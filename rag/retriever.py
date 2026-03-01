class Retriever:
    def __init__(self, embedding_model, vector_store, top_k=5, score_threshold=0.25):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query):
        query_vec = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=self.top_k)
        filtered = [(chunk, score) for chunk, score in results if score >= self.score_threshold]
        if not filtered:
            filtered = results[:1]
        return filtered

    def get_context(self, query):
        results = self.retrieve(query)
        parts = []
        sources = []
        for i, (chunk, score) in enumerate(results, 1):
            parts.append(f"[Section {i} | Page {chunk.page_number}]\n{chunk.text}")
            sources.append({
                "chunk_index": chunk.chunk_index,
                "page": chunk.page_number,
                "score": round(score, 4),
                "text": chunk.text
            })
        return "\n\n---\n\n".join(parts), sources
