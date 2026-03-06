"""
Embedding generation using Azure OpenAI text-embedding-3-small.
Replaces Doc2Vec + Word2Vec from main_2020.py.
"""

from openai import AzureOpenAI
from backend.config import settings
import numpy as np


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using Azure OpenAI.
    Returns numpy array of shape (len(texts), 1536).
    Handles batching for large lists.
    """
    client = get_client()
    deployment = settings.azure_openai_embedding_deployment

    # Azure OpenAI supports up to 2048 inputs per batch
    BATCH_SIZE = 500
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Clean empty strings
        batch = [t if t.strip() else "empty" for t in batch]

        response = client.embeddings.create(
            input=batch,
            model=deployment,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single text. Returns 1D array of shape (1536,)."""
    result = embed_texts([text])
    return result[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix from embeddings.
    Replaces WMD distance matrix from main_2020.py.
    Returns matrix of shape (n, n) where values are 1 - cosine_similarity.
    """
    # Normalize all vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Cosine similarity matrix
    sim_matrix = np.dot(normalized, normalized.T)

    # Convert to distance
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    return dist_matrix
