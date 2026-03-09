from src.document_loader import load_documents

def main() -> None:
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s).\n")

    for doc in documents:
        preview = doc["text"][:300].replace("\n", " ")
        print(f"Source: {doc['source']}")
        print(f"Preview: {preview}")
        print("-" * 80)

if __name__ == "__main__":
    main()