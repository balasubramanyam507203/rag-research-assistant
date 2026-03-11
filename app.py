import streamlit as st

from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.rag_pipeline import generate_rag_answer


st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def setup_vector_store() -> dict:
    """
    Load documents, chunk them, generate embeddings,
    and build the FAISS vector store once.
    """
    documents = load_documents()
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
    st.write("Ask questions about your research PDFs using Multi-Query Hybrid RAG.")

    try:
        data = setup_vector_store()
    except Exception as error:
        st.error(f"Error while preparing vector store: {error}")
        return

    st.sidebar.header("System Status")
    st.sidebar.write(f"Documents loaded: {len(data['documents'])}")
    st.sidebar.write(f"Chunks created: {len(data['chunks'])}")
    st.sidebar.write(f"Embeddings generated: {len(data['embeddings'])}")
    st.sidebar.write("Retrieval mode: Multi-Query Hybrid + Reranking")

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


if __name__ == "__main__":
    main()