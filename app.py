from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.retriever import prepare_rag_context

def main() -> None:
    documents = load_documents()
    chunked_documents = chunk_documents(documents)
    chunks, embeddings = generate_embeddings(chunked_documents)

    build_and_save_vector_store(chunks,embeddings)

    query = "What is few-shot tabular classification?"
    rag_data = prepare_rag_context(query=query, top_k=3)


    print(f"Loaded {len(documents)} documents(s).")
    print(f"Created {len(chunked_documents)} chunk(s).\n")
    print(f"Generated {len(embeddings)} embedding(s).\n")
    print("FAISS index created and saved.\n")

    print(f"Query: {rag_data['query']}\n")
    print("Retrieved chunks:\n")

    for chunk in rag_data["retrieved_chunks"]:
        preview = chunk["text"][:180].replace("\n", " ")
        print(f"Rank        : {chunk['rank']}")
        print(f"Distance    : {chunk['distance']:.4f}")
        print(f"Source      : {chunk['source']}")
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Preview     : {preview}")
        print("-" * 80)

    print("\nFormatted context for LLM:\n")
    print(rag_data["context"][:1200])
    


if __name__ == "__main__":
    main()