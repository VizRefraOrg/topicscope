"""
Distance matrix computation, PCA dimension reduction, force repulsion, and spacing.
"""

from sklearn.decomposition import PCA
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix
import numpy as np


def force_directed_repulsion(candidates, iterations=300, min_distance=0.04, repulsion=0.0005, damping=0.93, center_pull=0.0005):
    """
    Push overlapping topics apart while preserving relative positions.
    Same algorithm as v2 prototype.
    Creates water channels between topic islands.
    """
    pts = [{"x": c["x"], "y": c["y"], "vx": 0, "vy": 0, "size": c.get("size", 0.05)} for c in candidates]

    for _ in range(iterations):
        for p in pts:
            p["fx"] = 0
            p["fy"] = 0

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i]["x"] - pts[j]["x"]
                dy = pts[i]["y"] - pts[j]["y"]
                dist = max(np.sqrt(dx * dx + dy * dy), 0.001)
                min_d = min_distance + (pts[i]["size"] + pts[j]["size"]) * 0.8

                if dist < min_d:
                    force = repulsion * (min_d - dist) / dist
                    fx = dx * force
                    fy = dy * force
                    pts[i]["fx"] += fx
                    pts[i]["fy"] += fy
                    pts[j]["fx"] -= fx
                    pts[j]["fy"] -= fy

        for p in pts:
            p["fx"] -= p["x"] * center_pull
            p["fy"] -= p["y"] * center_pull
            p["vx"] = (p["vx"] + p["fx"]) * damping
            p["vy"] = (p["vy"] + p["fy"]) * damping
            p["x"] += p["vx"]
            p["y"] += p["vy"]

    for i, c in enumerate(candidates):
        c["x"] = pts[i]["x"]
        c["y"] = pts[i]["y"]

    return candidates


def compute_distance_and_reduce(
    candidates: list[dict],
    article_text: str,
    spacing_multiplier: float = 2.8,
    radius_multiplier: float = 2.2,
) -> list[dict]:
    """
    1. Embed all candidate titles
    2. Build cosine distance matrix (replaces WMD)
    3. PCA to 2D
    4. Force-directed repulsion to spread topics apart
    5. Apply spacing multiplier
    6. Compute sizes and heights
    """
    if len(candidates) < 2:
        for i, c in enumerate(candidates):
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = c.get("similarity", 0.5) * 0.06 * radius_multiplier
            c["height"] = c.get("similarity", 0.5)
        return candidates

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)

    dist_matrix = cosine_distance_matrix(embeddings)

    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(dist_matrix)
    coords = pca.components_

    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    max_sim = max(c.get("similarity", 0.01) for c in candidates)

    for i, candidate in enumerate(candidates):
        sim = candidate.get("similarity", 0.01)
        relevance = candidate.get("tfidf_relevance", sim)

        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier
        candidate["size"] = float(sim * 0.06 * radius_multiplier)
        candidate["height"] = float(max(sim, relevance))

    # ── FORCE REPULSION — push overlapping topics apart ──────────
    candidates = force_directed_repulsion(
        candidates,
        iterations=400,
        min_distance=0.06,
        repulsion=0.0006,
        damping=0.93,
        center_pull=0.0005,
    )

    return candidates
