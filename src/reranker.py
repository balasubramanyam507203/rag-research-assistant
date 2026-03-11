from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """
    Load and return the corss-enoder reranker model.
    """
    return CrossEncoder(model_name)

def rerank_chunks(query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using a CrossEncoder model.
    """
    if not retrieved_chunks:
        return[]
    
    reranker = load_reranker()

    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    reranked = []
    for chunk, score in zip(retrieved_chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["reranker_score"] = float(score)
        reranked.append(chunk_copy)

    reranked.sort(key=lambda item: item["reranker_score"], reverse=True)

    final_results = []
    for rank, chunk in enumerate(reranked[:top_k], start=1):
        chunk["rerank_rank"] = rank
        final_results.append(chunk)

    return final_results