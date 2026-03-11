from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY
from src.retriever import prepare_rag_context


def generate_rag_answer(
    query: str,
    top_k_dense: int = 5,
    top_k_bm25: int = 5,
    top_k_final: int = 3,
) -> Dict[str, Any]:
    """
    Full hybrid + reranked RAG pipeline:
    1. Retrieve dense + BM25 candidates
    2. Rerank them with a CrossEncoder
    3. Format context
    4. Send context + query to LLM
    5. Return answer with retrieved sources
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

    rag_data = prepare_rag_context(
        query=query,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        top_k_final=top_k_final,
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    prompt = f"""
You are a helpful research assistant.

Answer the user's question using only the provided context.
If the answer is not contained in the context, say:
"I could not find the answer in the provided documents."

Context:
{rag_data["context"]}

Question:
{query}
""".strip()

    response = llm.invoke(prompt)

    return {
        "query": query,
        "answer": response.content,
        "retrieved_chunks": rag_data["retrieved_chunks"],
        "context": rag_data["context"],
    }