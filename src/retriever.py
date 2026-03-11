from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from src.embeddings import load_embedding_model
from src.vector_store import load_faiss_index_and_metadata
from src.reranker import rerank_chunks


def retrieve_dense_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Dense retrieval using FAISS and embeddings.
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
            chunk_data["retrieval_type"] = "dense"
            results.append(chunk_data)

    return results


def retrieve_bm25_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Sparse / lexical retrieval using BM25.
    """
    _, metadata = load_faiss_index_and_metadata()

    tokenized_corpus = [chunk["text"].lower().split() for chunk in metadata]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(enumerate(scores))
    scored_chunks.sort(key=lambda item: item[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(scored_chunks[:top_k], start=1):
        chunk_data = metadata[idx].copy()
        chunk_data["rank"] = rank
        chunk_data["bm25_score"] = float(score)
        chunk_data["retrieval_type"] = "bm25"
        results.append(chunk_data)

    return results


def hybrid_retrieve_chunks(query: str, top_k_dense: int = 5, top_k_bm25: int = 5) -> List[Dict[str, Any]]:
    """
    Combine dense retrieval and BM25 retrieval, then deduplicate by chunk_id.
    """
    dense_results = retrieve_dense_chunks(query=query, top_k=top_k_dense)
    bm25_results = retrieve_bm25_chunks(query=query, top_k=top_k_bm25)

    merged: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for result in dense_results + bm25_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in seen_chunk_ids:
            merged.append(result)
            seen_chunk_ids.add(chunk_id)

    return merged


def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a single formatted context string for the LLM.
    """
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']} | Retrieval: {chunk['retrieval_type']}]\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def prepare_rag_context(
    query: str,
    top_k_dense: int = 5,
    top_k_bm25: int = 5,
    top_k_final: int = 3,
) -> Dict[str, Any]:
    """
    Hybrid retrieval + reranking pipeline:
    - retrieve dense chunks
    - retrieve BM25 chunks
    - merge and deduplicate
    - rerank with CrossEncoder
    - format into one context string
    """
    retrieved_chunks = hybrid_retrieve_chunks(
        query=query,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
    )

    reranked_chunks = rerank_chunks(
        query=query,
        retrieved_chunks=retrieved_chunks,
        top_k=top_k_final,
    )

    context = format_retrieved_context(reranked_chunks)

    return {
        "query": query,
        "retrieved_chunks": reranked_chunks,
        "context": context,
    }