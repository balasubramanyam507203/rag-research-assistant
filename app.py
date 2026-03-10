from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.rag_pipeline import generate_rag_answer

def main() -> None:
    documents = load_documents()
    chunked_documents = chunk_documents(documents)
    chunks, embeddings = generate_embeddings(chunked_documents)

    build_and_save_vector_store(chunks,embeddings)

    query = "What is few-shot tabular classification?"
    result = generate_rag_answer(query=query, top_k=3)


    print(f"Loaded {len(documents)} documents(s).")
    print(f"Created {len(chunked_documents)} chunk(s).\n")
    print(f"Generated {len(embeddings)} embedding(s).\n")
    print("FAISS index created and saved.\n")

    print(f"Question: {result['query']}\n")
    print("Answer:\n")
    print(result["answer"])
    print("\n" + "=" * 80)
    print("Sources used:\n")

    for chunk in result["retrieved_chunks"]:
        
        print(f"Rank        : {chunk['rank']}")
        print(f"Distance    : {chunk['distance']:.4f}")
        print(f"Source      : {chunk['source']}")
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print("-" * 80)

   
if __name__ == "__main__":
    main()