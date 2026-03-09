from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import generate_embeddings


def main() -> None:
    documents = load_documents()
    chunked_documents = chunk_documents(documents)
    chunks, embeddings = generate_embeddings(chunked_documents)


    print(f"Loaded {len(documents)} documents(s).")
    print(f"Created {len(chunked_documents)} chunk(s).\n")
    print(f"Generated {len(embeddings)} embedding(s).\n")

    if embeddings:
        print(f"Embedding dimension: {len(embeddings[0])}\n")

    for chunk, embedding in zip(chunks[:3], embeddings[:3]):
        preview = chunk["text"][:250].replace("\n", " ")
        print(f"Source  : {chunk['source']}")
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Preview : {preview}")
        print(f"Vector sample: {embedding[:8]}")
        print("-" * 80)

if __name__ == "__main__":
    main()