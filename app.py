from pathlib import Path

import streamlit as st
import pandas as pd

from src.config import DATA_DIR
from src.document_loader import load_documents, get_pdf_files
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.rag_pipeline import generate_rag_answer
from src.evaluator import evaluate_rag_system


st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")


def save_uploaded_files(uploaded_files) -> list[Path]:
    """
    Save uploaded PDF files into the data directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def setup_vector_store_from_documents(pdf_paths: list[Path]) -> dict:
    """
    Load documents, chunk them, generate embeddings,
    and build the FAISS vector store.
    """
    documents = load_documents(pdf_paths)
    chunked_documents = chunk_documents(documents)
    chunks, embeddings = generate_embeddings(chunked_documents)
    build_and_save_vector_store(chunks, embeddings)

    return {
        "documents": documents,
        "chunks": chunked_documents,
        "embeddings": embeddings,
    }


def main() -> None:
    st.title("📚 RAG Research Assistant")
    st.write("Upload research PDFs and ask questions using Multi-Query Hybrid RAG.")

    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    data = None

    if uploaded_files:
        if st.sidebar.button("Process Uploaded PDFs"):
            try:
                saved_paths = save_uploaded_files(uploaded_files)
                data = setup_vector_store_from_documents(saved_paths)
                st.session_state["data"] = data
                st.session_state["docs_ready"] = True
                st.sidebar.success("Uploaded PDFs processed successfully.")
            except Exception as error:
                st.sidebar.error(f"Error while processing uploaded PDFs: {error}")
    else:
        existing_files = get_pdf_files()
        if existing_files:
            if "docs_ready" not in st.session_state:
                try:
                    data = setup_vector_store_from_documents(existing_files)
                    st.session_state["data"] = data
                    st.session_state["docs_ready"] = True
                except Exception as error:
                    st.error(f"Error while loading default documents: {error}")
                    return

    if "data" in st.session_state:
        data = st.session_state["data"]

    if not data:
        st.info("Please upload PDF files from the sidebar to start asking questions.")
        return

    st.sidebar.header("System Status")
    st.sidebar.write(f"Documents loaded: {len(data['documents'])}")
    st.sidebar.write(f"Chunks created: {len(data['chunks'])}")
    st.sidebar.write(f"Embeddings generated: {len(data['embeddings'])}")
    st.sidebar.write("Retrieval mode: Multi-Query Hybrid + Reranking")

    st.subheader("Loaded Documents")
    for doc in data["documents"]:
        st.write(f"- {doc['source']}")

    query = st.text_input("Enter your question:")

    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating answer..."):
            try:
                result = generate_rag_answer(
                    query=query,
                    top_k_dense=5,
                    top_k_bm25=5,
                    top_k_final=3,
                    num_variations=3,
                )

                st.subheader("Expanded Queries")
                for expanded_query in result["expanded_queries"]:
                    st.write(f"- {expanded_query}")

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Final Reranked Sources")
                for chunk in result["retrieved_chunks"]:
                    title = (
                        f"Rerank #{chunk['rerank_rank']} | "
                        f"{chunk['retrieval_type'].upper()} | "
                        f"{chunk['source']} | "
                        f"{chunk['chunk_id']}"
                    )
                    with st.expander(title):
                        st.write(f"**Matched Query:** {chunk.get('matched_query', 'original')}")
                        st.write(f"**Reranker Score:** {chunk['reranker_score']:.4f}")
                        if "distance" in chunk:
                            st.write(f"**Dense Distance:** {chunk['distance']:.4f}")
                        if "bm25_score" in chunk:
                            st.write(f"**BM25 Score:** {chunk['bm25_score']:.4f}")
                        st.write(chunk["text"])

            except Exception as error:
                st.error(f"Error while generating answer: {error}")

    st.subheader("Evaluation")
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                eval_df = evaluate_rag_system()
                st.success("Evaluation completed.")
                st.dataframe(eval_df)
            except Exception as error:
                st.error(f"Error while running evaluation: {error}")


if __name__ == "__main__":
    main()