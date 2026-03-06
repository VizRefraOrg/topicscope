"""
Distance matrix computation and PCA dimension reduction.
Replaces WMD + PCA from main_2020.py.
"""

from sklearn.decomposition import PCA
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix
import numpy as np


def compute_distance_and_reduce(
    candidates: list[dict],
    article_text: str,
    spacing_multiplier: float = 2.8,
    radius_multiplier: float = 2.2,
) -> list[dict]:
    """
    1. Embed all candidate titles
    2. Build cosine distance matrix (replaces WMD)
    3. PCA to 2D (same as original)
    4. Apply spacing multiplier
    5. Compute sizes and heights

    Returns candidates with x, y, size, height added.
    """
    if len(candidates) < 2:
        # Not enough for PCA
        for i, c in enumerate(candidates):
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = c.get("similarity", 0.5) * 0.06 * radius_multiplier
            c["height"] = c.get("similarity", 0.5)
        return candidates

    # Embed all candidate titles
    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)

    # Cosine distance matrix (replaces WMD)
    dist_matrix = cosine_distance_matrix(embeddings)

    # PCA to 2D — same as original: pca.fit(phi); mds5 = pca.components_
    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(dist_matrix)
    coords = pca.components_

    # If only 1 component, pad with zeros
    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    # Assign positions with spacing multiplier
    max_sim = max(c.get("similarity", 0.01) for c in candidates)

    for i, candidate in enumerate(candidates):
        sim = candidate.get("similarity", 0.01)
        relevance = candidate.get("tfidf_relevance", sim)

        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier

        # Size based on similarity score (same logic as original: circle_sizes = sim * 0.06)
        candidate["size"] = float(sim * 0.06 * radius_multiplier)

        # Height based on relevance (used for Z-axis in 3D)
        candidate["height"] = float(max(sim, relevance))

    return candidates
