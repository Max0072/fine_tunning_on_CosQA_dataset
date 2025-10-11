## Project Structure

```
.
├── MNRL_fine_tuning.py       # Main training script with MNRL loss
├── evaluation.py              # Evaluation metrics (Recall, MRR, NDCG)
├── search_engine.py           # FAISS-based search engine implementation
├── API.py                     # FastAPI server for search engine
├── API_README.md              # API documentation
├── requirements.txt           # Project dependencies
├── best_model/                # Fine-tuned model weights and config
└── test_data_for_api/        # Sample data for API testing
```

## Core Components

### 1. Training (MNRL_fine_tuning.py)
- Implements Multiple Negatives Ranking Loss for contrastive learning
- Uses in-batch negatives: for each query-code pair, all other codes in the batch serve as negatives

### 2. Search Engine (search_engine.py)
- FAISS-based vector similarity search with `IndexFlatIP` (Inner Product)
- Supports:
  - Document upload and indexing
  - Single and batch query search

### 3. Evaluation (evaluation.py)
- Implements three retrieval metrics:
  - **Recall@10**: Percentage of queries with relevant result in top 10
  - **MRR@10**: Mean Reciprocal Rank of the first relevant result
  - **NDCG@10**: Normalized Discounted Cumulative Gain
- Evaluates model on CosQA test set

### 4. API (API.py)
- FastAPI-based REST API for the search engine
- Endpoints:
  - `POST /upload_json`: Upload documents (JSON/JSONL)
  - `POST /search`: Single query search
  - `POST /search_batch`: Batch query search
  - `DELETE /delete_all`: Clear all indexed data
- See [API_README.md](API_README.md) for detailed usage

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python MNRL_fine_tuning.py
```

This will:
1. Load the CosQA dataset
2. Fine-tune the model
3. Save the best model to `./best_model/`
4. Display training metrics and plot training loss

### Evaluating the Model

```bash
python evaluation.py
```

### Using the Search Engine (Python API)

```python
from sentence_transformers import SentenceTransformer
from search_engine import SearchEngine

# Load fine-tuned model
model = SentenceTransformer("best_model")
engine = SearchEngine(model=model)

# Upload documents
documents = [
    {"_id": "1", "text": "def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)"},
    {"_id": "2", "text": "function to calculate factorial recursively"}
]
engine.upload_data(documents)

# Search
results = engine.find_similar_from_text("fibonacci function", k=5)
print(results)  # {'ids': [...], 'scores': [...]}
```

### Using the FastAPI Server

In order for the API to work, you need to have the server running. You can start the server by executing:
```bash
python API.py
```

After starting the server, you can interact with the API using `curl` commands as shown below.

#### Uploading data
```bash
curl -X POST "http://0.0.0.0:8000/upload_json" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_data1.json"
```

#### Sending single queries
```bash
curl -X POST "http://0.0.0.0:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"text": "dog", "k": 3}'
```

#### Sending multiple queries
```bash
curl -X POST "http://0.0.0.0:8000/search_batch" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_query1.jsonl"
```

```bash
curl -X POST "http://0.0.0.0:8000/search_batch" \
  -H "Content-Type: application/json" \
  -d '[{"text": "dog", "k": 3}, {"text": "cat", "k": 3}]'
```

#### Deleting all data from index
```bash
curl -X DELETE "http://0.0.0.0:8000/delete_all" \
  -H "accept: application/json"
```



