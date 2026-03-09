from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load and return the sentence transformer embedding model.
    """
    return SentenceTransformer(model_name)

def generate_embeddings(
    chunked_documents: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[List[float]]]:
    """
    Generate embeddings from chunked documents.
    
    Returns:
        - original chunked documents
        - list of embeddings corresponding to each chunk
    """
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunked_documents]
    embeddings = model.encode(texts, convert_to_numpy=True)

    return chunked_documents, embeddings.tolist()