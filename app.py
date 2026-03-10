from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings
from src.vector_store import build_and_save_vector_store
from src.retriever import retrieve_similar_chunks

def main() -> None:
    documents = load_documents()
    chunked_documents = chunk_documents(documents)
    chunks, embeddings = generate_embeddings(chunked_documents)

    build_and_save_vector_store(chunks,embeddings)


    print(f"Loaded {len(documents)} documents(s).")
    print(f"Created {len(chunked_documents)} chunk(s).\n")
    print(f"Generated {len(embeddings)} embedding(s).\n")
    print("FAISS index created and saved.\n")

    query = "What is few-shot tabular classification?"
    results = retrieve_similar_chunks(query=query, top_k=3)

    print(f"Query: {query}\n")
    print("Top retrieved chunks:\n")

    for result in results:
        preview = result["text"][:220].replace("\n", " ")
        print(f"Source    :{result['source']}")
        print(f"Chunk ID  : {result['chunk_id']}")
        print(f"Preview   : {preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()