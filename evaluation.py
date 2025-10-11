from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from search_engine import SearchEngine
import numpy as np


def recall_10(ranks):
    return sum([1 if i > 0 else 0 for i in ranks]) / len(ranks)

def mrr_10(ranks):
    return sum([1/i if i > 0 else 0 for i in ranks]) / len(ranks)

def ndcg_10(ranks):
    return sum([1 / np.log2(i + 1) if i > 0 else 0 for i in ranks]) / len(ranks)

# outputs ranks (top10)
def evaluate_model_on_dataset(model):
    Engine = SearchEngine(model=model)
    ds_retrieval = load_dataset("CoIR-Retrieval/cosqa")
    ds_test = ds_retrieval["test"]

    ds = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    ds_corpus = ds["corpus"]
    test_corpus = ds_corpus.filter(lambda x: x["partition"] == "test")
    ds_queries = ds["queries"]
    test_queries = ds_queries.filter(lambda x: x["partition"] == "test")

    Engine.upload_data(test_corpus)                                             # upload corpus to the faiss index
    results = Engine.find_similar_from_text_batch(test_queries, 10)    # search for each query in the index
    # results have format: [{"ids": [...], "scores": [...]}, ...]

    ranks = []

    for i in range(ds_test.shape[0]):
        corp_id = ds_test[i]["corpus-id"]
        int_corp_id = Engine.str_id_to_int_id[corp_id]
        rank = results[i]["ids"].index(int_corp_id) + 1 if int_corp_id in results[i]["ids"] else -1
        ranks.append(rank)
    return ranks


def get_metrics(model):
    ranks = evaluate_model_on_dataset(model)
    return recall_10(ranks), mrr_10(ranks), ndcg_10(ranks)


def main():
    model = SentenceTransformer("all-miniLM-L6-v2")
    recall_10, mrr_10, ndcg_10 = get_metrics(model)
    print(f"Recall-10: {recall_10}")
    print(f"MRR-10: {mrr_10}")
    print(f"NDCG-10: {ndcg_10}")


if __name__ == "__main__":
    main()

