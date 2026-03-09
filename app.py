from src.document_loader import load_documents
from src.text_chunker import chunk_documents


def main() -> None:
    documents = load_documents()
    chunked_documents = chunk_documents(documents)

    print(f"Loaded {len(documents)} documents(s).")
    print(f"Created {len(chunked_documents)} chunk(s).\n")

    for chunk in chunked_documents[:5]:
        preview = chunk["text"][:250].replace("\n", " ")
        print(f"Source  : {chunk['source']}")
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Preview : {preview}")
        print("-" * 80)

if __name__ == "__main__":
    main()