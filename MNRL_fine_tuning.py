import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import matplotlib.pyplot as plt

from transformers import get_constant_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import math
from evaluation import get_metrics

train_step_losses = []

def plot_graph(steps_per_epoch, total_epochs):

    losses = np.array(train_step_losses)

    window = 25
    kernel = np.ones(window) / window
    moving = np.convolve(losses, kernel, mode="valid")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses, label="Batch loss (per step)")
    plt.plot(moving, label=f"Moving average (window={window})")

    for e in range(1, total_epochs):
        plt.axvline(e * steps_per_epoch, linestyle="--", linewidth=1)

    plt.title("Training Loss over Steps")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
    print("Plot saved as training_loss.png")
    plt.show()

# function for DataLoader
def collate_fn(batch):
    queries, corpus = zip(*batch)
    return list(queries), list(corpus)

# process data
def get_pairs(queries, corpus, scores):
    corpus_lookup = {c["_id"]: c["text"] for c in corpus}
    query_lookup = {q["_id"]: q["text"] for q in queries}
    pairs = []
    for row in scores:
        q_text = query_lookup[row["query-id"]]
        c_text = corpus_lookup[row["corpus-id"]]
        pairs.append((q_text, c_text))
    return pairs

# load the dataset
def get_dataset():
    scores_full = load_dataset("CoIR-Retrieval/cosqa")
    scores_train = scores_full["train"].filter(lambda x: x["score"] == 1)
    scores_val = scores_full["valid"]
    scores_test = scores_full["test"]

    ds = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    q_full = ds["queries"]
    q_train = q_full.filter(lambda x: x["partition"] == "train")
    q_val = q_full.filter(lambda x: x["partition"] == "valid")
    q_test = q_full.filter(lambda x: x["partition"] == "test")
    c_full = ds["corpus"]
    c_train = c_full.filter(lambda x: x["partition"] == "train")
    c_val = c_full.filter(lambda x: x["partition"] == "valid")
    c_test = c_full.filter(lambda x: x["partition"] == "test")

    train_pairs = get_pairs(q_train, c_train, scores_train)
    val_pairs = get_pairs(q_val, c_val, scores_val)
    test_pairs = get_pairs(q_test, c_test, scores_test)

    return train_pairs, val_pairs, test_pairs


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# from text to emb
def forward_embeddings(model, texts, device):
    features = model.tokenize(texts)
    for k in features:
        features[k] = features[k].to(device)
    emb = model(features)
    return emb["sentence_embedding"]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, temp, max_norm, device, epoch, num_epochs):
    global train_step_losses
    model.train()
    total_loss, n = 0.0, 0
    for queries, corpus in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
        optimizer.zero_grad(set_to_none=True)

        q = forward_embeddings(model, queries, device)
        c = forward_embeddings(model, corpus, device)
        logits = (q @ c.T) / temp
        y = torch.arange(logits.size(0), device=device)
        loss = criterion(logits, y) + criterion(logits.T, y)
        loss.backward()

        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        n += bs
        total_loss += loss.item() * bs
        train_step_losses.append(loss.item())

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, temp, device):
    model.eval()
    total_loss, n = 0, 0
    for queries, corpus in loader:

        q = forward_embeddings(model, queries, device)
        c = forward_embeddings(model, corpus, device)
        logits = (q @ c.T) / temp
        y = torch.arange(logits.size(0), device=device)
        loss = criterion(logits, y) + criterion(logits.T, y)

        bs = y.size(0)
        n += bs
        total_loss += loss.item() * bs

    return total_loss / n


def main():
    # set seed and get avaliable device
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # load dataset
    train_pairs, val_pairs, test_pairs = get_dataset()

    batch_size = 64
    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=False)

    # timings
    total_epochs = 10
    steps_per_epoch = len(train_loader)
    num_training_steps = total_epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * num_training_steps)

    # model and optimization parameters
    model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    model.max_seq_length = 128
    temp = 0.05
    max_norm = 1.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=1e-2)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)


    # initial loss (so we know that the model actually became better)
    initial_val_loss = evaluate(model, val_loader, criterion, temp, device)
    initial_test_loss = evaluate(model, test_loader, criterion, temp, device)
    print(f"Initial val loss: {initial_val_loss:.3f}")
    print(f"Initial test loss: {initial_test_loss:.3f}")

    # training loop
    best_model_path = "./best_model"
    best_val_loss = math.inf
    bad_epochs = 0
    patience = 2

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, temp, max_norm, device, epoch, total_epochs)
        val_loss = evaluate(model, val_loader, criterion, temp, device)

        print(f"Epoch {epoch:02d} | "
              f"Train_loss: {train_loss:.3f} | "
              f"Val_loss: {val_loss:.3f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(best_model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs > patience:
            print("Early stopping triggered")
            break

    print("Best val loss:", best_val_loss)

    model = SentenceTransformer(best_model_path, device=str(device))
    test_loss = evaluate(model, test_loader, criterion, temp, device)

    print(f"Test loss: {test_loss:.3f}")

    recall_10, mrr_10, ndcg_10 = get_metrics(model)
    print(f"Recall-10: {recall_10}")
    print(f"MRR-10: {mrr_10}")
    print(f"NDCG-10: {ndcg_10}")

    plot_graph(steps_per_epoch, total_epochs)


if __name__ == "__main__":
    main()


