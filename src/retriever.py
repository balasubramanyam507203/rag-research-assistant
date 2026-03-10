from typing import List, Dict, Any

import numpy as np

from src.embeddings import load_embedding_model
from src.vector_store import load_faiss_index_and_metadata

def retrieve_similar_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar chunks for a given query, including similarity distance.
    """
    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    index, metadata = load_faiss_index_and_metadata()
    distances, indices = index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []

    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < len(metadata):
            chunk_data = metadata[idx].copy()
            chunk_data["rank"] = rank
            chunk_data["distance"] = float(distance)
            results.append(chunk_data)

    return results

def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a single formatted context string for the LLM.
    """
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}]\n{chunk['text']}"

        )

    return "\n\n".join(context_parts)

def prepare_rag_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Full retrieval pipeline:
    - retrieve top-k chunks
    - format them into one context string
    - return both raw chunks and formatted context
    """
    retrieved_chunks = retrieve_similar_chunks(query=query, top_k=top_k)
    context = format_retrieved_context(retrieved_chunks)

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "context" : context,
    }