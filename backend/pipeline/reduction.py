"""
Distance matrix, PCA reduction, force repulsion, scaling.
Back to PCA (stable, deterministic) with strong force repulsion for water channels.
"""

from sklearn.decomposition import PCA
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np


def force_directed_repulsion(candidates, iterations=600, min_distance=0.12, repulsion=0.002, damping=0.90, center_pull=0.0002):
    """
    Strong force repulsion to create water channels between islands.
    Higher min_distance and repulsion than before.
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

                # Minimum distance considers BOTH topics' sizes
                # so bigger topics push further apart
                min_d = min_distance + (pts[i]["size"] + pts[j]["size"]) * 1.5

                if dist < min_d:
                    force = repulsion * (min_d - dist) / dist
                    pts[i]["fx"] += dx * force
                    pts[i]["fy"] += dy * force
                    pts[j]["fx"] -= dx * force
                    pts[j]["fy"] -= dy * force

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
    entities_text: str = "",
    spacing_multiplier: float = 3.0,
    radius_multiplier: float = 2.2,
) -> dict:
    """
    1. Embed all candidate titles
    2. Cosine distance matrix
    3. PCA to 2D
    4. Compute size (from confidence) and height (from relevance)
    5. Strong force repulsion for water channels
    6. Return candidates + debug data
    """
    if len(candidates) < 2:
        for c in candidates:
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = 0.05 * radius_multiplier
            c["height"] = c.get("similarity", 0.5)
        return {"candidates": candidates, "distance_matrix": [], "debug": []}

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    # Embed article and entities for scoring
    article_emb = embed_texts([article_text])[0]
    entities_emb = embed_texts([entities_text])[0] if entities_text.strip() else None

    # ── PCA to 2D ────────────────────────────────────────────────
    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(dist_matrix)
    coords = pca.components_

    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    # ── Compute raw scores ───────────────────────────────────────
    POWER = 2.5  # Power curve to stretch narrow similarity bands

    debug_info = []

    for i, candidate in enumerate(candidates):
        sim_article = float(cosine_similarity(embeddings[i], article_emb))
        sim_entities = float(cosine_similarity(embeddings[i], entities_emb)) if entities_emb is not None else 0.0
        tfidf_rel = candidate.get("tfidf_relevance", 0.0)

        # Size = confidence score (determines x-y footprint of half-ellipse)
        # Use similarity to entities as primary, article as fallback
        raw_confidence = max(sim_entities, sim_article * 0.6, tfidf_rel)

        # Height = relevance (determines z-axis of half-ellipse)
        raw_height = max(sim_entities * 0.8, sim_article * 0.5, tfidf_rel)

        candidate["_raw_confidence"] = raw_confidence
        candidate["_raw_height"] = raw_height
        candidate["_sim_article"] = sim_article
        candidate["_sim_entities"] = sim_entities
        candidate["_tfidf"] = tfidf_rel

    # ── Power-curve stretch and normalize ────────────────────────
    raw_confs = np.array([c["_raw_confidence"] for c in candidates])
    raw_heights = np.array([c["_raw_height"] for c in candidates])

    confs_stretched = np.power(np.clip(raw_confs, 0.001, 1.0), POWER)
    heights_stretched = np.power(np.clip(raw_heights, 0.001, 1.0), POWER)

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.ones_like(arr) * 0.5

    confs_norm = norm01(confs_stretched)
    heights_norm = norm01(heights_stretched)

    # ── Assign positions and values ──────────────────────────────
    for i, candidate in enumerate(candidates):
        # Position from PCA, scaled
        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier

        # Size = half-ellipse x-y footprint radius
        # Range: 0.02 (tiny islet) to 0.09 (dominant island) before radius_multiplier
        candidate["size"] = float(0.02 + confs_norm[i] * 0.07) * radius_multiplier

        # Height = half-ellipse z-axis peak
        # 0 to 1 normalized (frontend maps to v3 range)
        candidate["height"] = float(heights_norm[i])

        debug_info.append({
            "title": candidate["title"],
            "source": candidate.get("source", ""),
            "sim_article": round(candidate["_sim_article"], 4),
            "sim_entities": round(candidate["_sim_entities"], 4),
            "tfidf": round(candidate["_tfidf"], 4),
            "raw_confidence": round(candidate["_raw_confidence"], 4),
            "raw_height": round(candidate["_raw_height"], 4),
            "final_height": round(candidate["height"], 4),
            "final_size": round(candidate["size"], 4),
            "x": round(candidate["x"], 4),
            "y": round(candidate["y"], 4),
        })

    # ── Force repulsion — strong, creates water channels ─────────
    candidates = force_directed_repulsion(
        candidates,
        iterations=600,
        min_distance=0.12,
        repulsion=0.002,
        damping=0.90,
        center_pull=0.0002,
    )

    # Update debug with post-repulsion positions
    for i, d in enumerate(debug_info):
        d["x_final"] = round(candidates[i]["x"], 4)
        d["y_final"] = round(candidates[i]["y"], 4)

    # Build serializable distance matrix
    dm_list = []
    topic_labels = [c["title"] for c in candidates]
    for i in range(len(candidates)):
        row = {}
        for j in range(len(candidates)):
            row[topic_labels[j]] = round(float(dist_matrix[i][j]), 4)
        dm_list.append({"topic": topic_labels[i], "distances": row})

    # Clean temp fields
    for c in candidates:
        for k in ["_raw_confidence", "_raw_height", "_sim_article", "_sim_entities", "_tfidf"]:
            c.pop(k, None)

    return {
        "candidates": candidates,
        "distance_matrix": dm_list,
        "debug": debug_info,
    }
