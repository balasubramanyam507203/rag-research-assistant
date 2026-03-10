import streamlit as st

from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.rag_pipeline import generate_rag_answer


st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def setup_vector_store() -> dict:
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
    st.write("Ask questions about your research PDFs using Retrieval-Augmented Generation.")

    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
        st.session_state.system_data = None

    if not st.session_state.system_ready:
        if st.button("Initialize RAG System"):
            with st.spinner("Loading documents, generating embeddings, and building FAISS index..."):
                try:
                    st.session_state.system_data = setup_vector_store()
                    st.session_state.system_ready = True
                    st.success("RAG system initialized successfully.")
                except Exception as error:
                    st.error(f"Initialization failed: {error}")
                    return
        return

    data = st.session_state.system_data

    st.sidebar.header("System Status")
    st.sidebar.write(f"Documents loaded: {len(data['documents'])}")
    st.sidebar.write(f"Chunks created: {len(data['chunks'])}")
    st.sidebar.write(f"Embeddings generated: {len(data['embeddings'])}")

    query = st.text_input("Enter your question:")

    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating answer..."):
            try:
                result = generate_rag_answer(query=query, top_k=3)

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources Used")
                for chunk in result["retrieved_chunks"]:
                    with st.expander(
                        f"Rank {chunk['rank']} | {chunk['source']} | {chunk['chunk_id']}"
                    ):
                        st.write(f"**Distance:** {chunk['distance']:.4f}")
                        st.write(chunk["text"])

            except Exception as error:
                st.error(f"Error while generating answer: {error}")


if __name__ == "__main__":
    main()