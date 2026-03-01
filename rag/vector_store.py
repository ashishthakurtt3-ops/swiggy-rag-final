import os
import json
import pickle
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_chunks(self, chunks, embeddings):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec, top_k=5):
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"dimension": self.dimension}, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        store = cls(dimension=meta["dimension"])
        store.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store
