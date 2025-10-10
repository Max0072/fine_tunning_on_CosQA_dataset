import faiss
import numpy as np
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datasets import Dataset

import os, faiss

def faiss_threads_to_one():
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)


# Data class
class SEjsonDataclass(BaseModel):
    id: str = Field(default=None, alias="_id")
    partition: Optional[str] = None
    text: str
    title: Optional[str] = None
    language: Optional[str] = None
    meta_information: Optional[Dict[str, Any]] = None


class SearchEngine:
    def __init__(self, model):
        self.model = model
        dimension = 384
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self.str_id_to_int_id: Dict[str, int] = {}
        self.int_id_to_data: Dict[int, Any] = {}
        self.data: List[Any] = []
        self._next_int_id = 0
        faiss_threads_to_one() # due to the segfaults in OpenMP on macOS

    # updates the model
    def update_model(self, model):
        self.model = model

    # Get an embedding with normalization
    def embed_normalized(self, text: str) -> np.ndarray:
        self.model.eval()
        embedding = self.model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm != 0.0 else embedding
        return normalized_embedding.astype(np.float32)

    # Checks if the index is empty (If it's empty, no search can be performed)
    def index_is_empty(self) -> bool:
        return self.index.ntotal == 0

    # Delete all data and reset the index
    def delete_all_data(self):
        self.data.clear()
        self.str_id_to_int_id.clear()
        self.int_id_to_data.clear()
        self._next_int_id = 0
        self.index.reset()
        print("All data deleted.")

    # Add a list of texts to the index
    def add_corpus_to_index(self, corpus: List[str]):
        embeddings = [self.embed_normalized(text) for text in corpus]
        ids = np.arange(len(self.data) - len(corpus), len(self.data))
        self.index.add_with_ids(np.array(embeddings).astype(np.float32), ids.astype(np.int64))
        # print(f"Vectors in the index: {self.index.ntotal}")

    # Upload data from a list of dicts or a Dataset
    def upload_data(self, rows: List[dict] | Dataset):
        for row in rows:
            doc = SEjsonDataclass(**row)
            self.data.append(doc)
            str_id = doc.id or str(self._next_int_id)
            self.str_id_to_int_id[str_id] = self._next_int_id
            self.int_id_to_data[self._next_int_id] = doc
            self._next_int_id += 1
            self.add_corpus_to_index([doc.text])
        print("Data updated.")

    # Find similar items from a given vector
    def find_similar_from_vector(self, vec: np.ndarray, k: int):
        vec = vec.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(vec, k)
        return scores[0], ids[0]

    # Find similar items from a given text
    def find_similar_from_text(self, text: str, k: int) -> Dict[str, List]:
        vec = self.embed_normalized(text.lower())
        scores, ids = self.find_similar_from_vector(vec, k)
        return {"ids": ids.tolist(), "scores": scores.tolist()}

    # Find similar items from a batch of texts
    def find_similar_from_text_batch(self, rows: List[Dict] | Dataset, default_k: int = 5) -> List[Dict[str, List]]:
        results = []
        for r in rows:
            text = r.get("text")
            k = r.get("k", default_k)
            result = self.find_similar_from_text(text, k)
            results.append(result)
        return results
