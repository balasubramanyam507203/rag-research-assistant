from typing import List, Dict

from langchain_text_splitters  import RecursiveCharacterTextSplitter


def chunk_documents(
        documents: List[Dict[str, str]],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Split loaded documents into smaller chunks while preserving source metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_documents = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        chunks = text_splitter.split_text(text)

        for index, chunk in enumerate(chunks):
            chunked_documents.append(
                {
                    "source": source,
                    "chunk_id": f"{source}_chunk_{index}",
                    "text": chunk,
                }
            )
    
    return chunked_documents

