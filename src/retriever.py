from typing import List, Dict

import numpy as np

from src.embeddings import load_embedding_model
from src.vector_store import load_faiss_index_and_metadata

def retrieve_similar_chunks(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve the most similar chunks for a given query.
    """
    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    index, metadata = load_faiss_index_and_metadata()
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return results