"""
Distance matrix, t-SNE reduction, force repulsion, power-curve scaling.
"""

from sklearn.manifold import TSNE
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np


def force_directed_repulsion(candidates, iterations=500, min_distance=0.08, repulsion=0.001, damping=0.92, center_pull=0.0003):
    """Push overlapping topics apart. Creates water channels between islands."""
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
                min_d = min_distance + (pts[i]["size"] + pts[j]["size"]) * 1.2

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
    spacing_multiplier: float = 2.8,
    radius_multiplier: float = 2.2,
) -> dict:
    """
    Returns dict with:
      - candidates: list with x, y, size, height added
      - distance_matrix: raw cosine distance matrix
      - debug: per-candidate debug info
    """
    if len(candidates) < 2:
        for c in candidates:
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = c.get("similarity", 0.5) * 0.06 * radius_multiplier
            c["height"] = c.get("similarity", 0.5)
        return {"candidates": candidates, "distance_matrix": [], "debug": []}

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    # Also embed entities text for height calculation
    if entities_text.strip():
        entities_emb = embed_texts([entities_text])[0]
    else:
        entities_emb = None

    # Also embed the article
    article_emb = embed_texts([article_text])[0]

    # ── t-SNE ────────────────────────────────────────────────────
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

    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs

    # ── Compute raw scores ───────────────────────────────────────
    POWER = 3.0

    debug_info = []

    for i, candidate in enumerate(candidates):
        sim_article = float(cosine_similarity(embeddings[i], article_emb))
        sim_entities = float(cosine_similarity(embeddings[i], entities_emb)) if entities_emb is not None else 0.0
        tfidf_rel = candidate.get("tfidf_relevance", 0.0)

        # Height based on similarity to ENTITIES (not full article)
        # This matches your original algorithm: relevance = cosine_sim(title, entities)
        # Falls back to article sim if no entities
        raw_height = max(sim_entities, sim_article * 0.5, tfidf_rel)

        candidate["_sim_article"] = sim_article
        candidate["_sim_entities"] = sim_entities
        candidate["_tfidf"] = tfidf_rel
        candidate["_raw_height"] = raw_height
        candidate["_raw_sim"] = candidate.get("similarity", 0.0)

    # ── Power-curve stretch ──────────────────────────────────────
    raw_sims = np.array([c["_raw_sim"] for c in candidates])
    raw_heights = np.array([c["_raw_height"] for c in candidates])

    sims_stretched = np.power(np.clip(raw_sims, 0.001, 1.0), POWER)
    heights_stretched = np.power(np.clip(raw_heights, 0.001, 1.0), POWER)

    # Normalize to 0-1
    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5

    sims_norm = norm01(sims_stretched)
    heights_norm = norm01(heights_stretched)

    # ── Assign positions, sizes, heights ─────────────────────────
    for i, candidate in enumerate(candidates):
        candidate["x"] = float(coords[i, 0]) * spacing_multiplier
        candidate["y"] = float(coords[i, 1]) * spacing_multiplier

        # Size: 0.02 to 0.09 range (before radius multiplier)
        candidate["size"] = float(0.02 + sims_norm[i] * 0.07) * radius_multiplier

        # Height: 0 to 1 (will be normalized to v3 range on frontend)
        candidate["height"] = float(heights_norm[i])

        debug_info.append({
            "title": candidate["title"],
            "source": candidate.get("source", ""),
            "sim_article": round(candidate["_sim_article"], 4),
            "sim_entities": round(candidate["_sim_entities"], 4),
            "tfidf": round(candidate["_tfidf"], 4),
            "raw_height": round(candidate["_raw_height"], 4),
            "final_height": round(candidate["height"], 4),
            "final_size": round(candidate["size"], 4),
            "x": round(candidate["x"], 4),
            "y": round(candidate["y"], 4),
        })

    # ── Force repulsion — push apart for water channels ──────────
    candidates = force_directed_repulsion(
        candidates,
        iterations=500,
        min_distance=0.08,
        repulsion=0.001,
        damping=0.92,
        center_pull=0.0003,
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

    # Clean up temp fields
    for c in candidates:
        for k in ["_sim_article", "_sim_entities", "_tfidf", "_raw_height", "_raw_sim"]:
            c.pop(k, None)

    return {
        "candidates": candidates,
        "distance_matrix": dm_list,
        "debug": debug_info,
    }
