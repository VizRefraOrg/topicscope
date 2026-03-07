"""
Distance matrix computation, t-SNE dimension reduction, and scaling.
t-SNE replaces PCA because Azure OpenAI cosine distances are compressed
into a narrow band (0.05-0.25). t-SNE amplifies small differences into
large visual gaps, creating the island archipelago effect naturally.
"""

from sklearn.manifold import TSNE
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
    2. Build cosine distance matrix
    3. t-SNE to 2D (replaces PCA — amplifies narrow distance bands)
    4. Apply spacing multiplier
    5. Power-curve stretch for sizes and heights
    """
    if len(candidates) < 2:
        for c in candidates:
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = c.get("similarity", 0.5) * 0.06 * radius_multiplier
            c["height"] = c.get("similarity", 0.5)
        return candidates

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    # ── t-SNE dimension reduction ────────────────────────────────
    # perplexity must be less than n_samples
    n = len(candidates)
    perplexity = min(10, max(2, n // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="precomputed",
        init="random",
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
    )
    coords = tsne.fit_transform(dist_matrix)

    # Normalize coords to roughly -1 to 1 range before spacing
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs

    # ── Power-curve stretch for size and height ──────────────────
    # Raw similarity scores from Azure OpenAI cluster in a narrow band
    # (e.g. 0.45-0.59). Power curve amplifies small differences.
    #
    # score^3 turns:  0.45, 0.50, 0.55, 0.59
    # into:           0.091, 0.125, 0.166, 0.205  (2.25x spread vs 1.3x)
    POWER = 3.0

    sims = np.array([c.get("similarity", 0.01) for c in candidates])
    relevances = np.array([c.get("tfidf_relevance", c.get("similarity", 0.01)) for c in candidates])

    # Apply power curve
    sims_stretched = np.power(sims, POWER)
    rel_stretched = np.power(np.maximum(sims, relevances), POWER)

    # Normalize to 0-1 range
    sim_min, sim_max = sims_stretched.min(), sims_stretched.max()
    if sim_max > sim_min:
        sims_norm = (sims_stretched - sim_min) / (sim_max - sim_min)
    else:
        sims_norm = np.ones_like(sims_stretched) * 0.5

    rel_min, rel_max = rel_stretched.min(), rel_stretched.max()
    if rel_max > rel_min:
        rel_norm = (rel_stretched - rel_min) / (rel_max - rel_min)
    else:
        rel_norm = np.ones_like(rel_stretched) * 0.5

    # ── Assign values ────────────────────────────────────────────
    for i, candidate in enumerate(candidates):
        candidate["x"] = float(coords[i, 0]) * spacing_multiplier
        candidate["y"] = float(coords[i, 1]) * spacing_multiplier

        # Size: map from power-stretched similarity
        # Range matches v3: smallest ~0.02, largest ~0.09 (before radius mult)
        candidate["size"] = float(0.02 + sims_norm[i] * 0.07) * radius_multiplier

        # Height: map from power-stretched relevance
        # Range matches v3: 0.038 to 0.28
        candidate["height"] = float(rel_norm[i])

    return candidates
