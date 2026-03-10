import json
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np

from src.config import BASE_DIR


FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_FILE = FAISS_INDEX_DIR / "document_index.faiss"
METADATA_FILE = FAISS_INDEX_DIR / "document_metadata.json"


def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from embeddings.
    """
    embedding_array = np.array(embeddings, dtype="float32")
    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)

    return index

def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """
    Save the FAISS index to disk.
    """
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

def save_metadata(chunks: List[Dict[str, str]]) -> None:
    """
    Save chunk metadata to disk as JSON.
    """
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

def build_and_save_vector_store(
        chunks: List[Dict[str, str]],
        embeddings: List[List[float]],
) -> None:
    """
    Build FAISS index and save both index and metadata.
    """
    index = create_faiss_index(embeddings)
    save_faiss_index(index)
    save_metadata(chunks)

def load_faiss_index_and_metadata() -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]]]:
    """
    Load FAISS index and metadata from disk.
    """
    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return index, metadata