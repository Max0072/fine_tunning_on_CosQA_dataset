import json
from typing import List, Any
from fastapi import FastAPI, UploadFile, File, Request
from search_engine import SearchEngine
from sentence_transformers import SentenceTransformer


# This api allows uploading JSON/JSONL data and perform similarity search.
# Read the README_API.md for details.

app = FastAPI(title="VectorSearchAPI")

Engine = SearchEngine(model=SentenceTransformer("all-MiniLM-L6-v2"))  # Initialize the search engine (model and index)


# function to parse JSON or JSONL from bytes
def parse_json_or_jsonl(raw: bytes, encoding: str = 'utf-8') -> List[Any]:
    text = raw.decode(encoding).strip()
    if text.startswith('['):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON content is not a list")
    items = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


@app.post("/upload_json")
async def upload_json(request: Request, file: UploadFile | None = File(None)):
    if file:
        raw = await file.read()
        rows = parse_json_or_jsonl(raw)
    else:
        rows = await request.json()
    Engine.upload_data(rows)
    return {"loaded": len(rows)}


@app.delete("/delete_all")
def delete_all():
    Engine.delete_all_data()
    return {"status": "all data deleted"}


@app.post("/search")
def search(query: dict):
    if Engine.index_is_empty():
        return {"detail": "Index is empty. Upload data first."}
    text = query.get("text")
    k = query.get("k", 5)
    result = Engine.find_similar_from_text(text, k)
    return result


@app.post("/search_batch")
async def search_batch(request: Request, file: UploadFile | None = File(None)):
    if Engine.index_is_empty():
        return {"detail": "Index is empty. Upload data first."}
    if file:
        raw = await file.read()
        rows = parse_json_or_jsonl(raw)
    else:
        rows = await request.json()

    results = Engine.find_similar_from_text_batch(rows)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)