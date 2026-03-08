"""
Distance matrix, StandardScaler + PCA, force repulsion.

Key fix from notebook: StandardScaler BEFORE PCA.
This amplifies small cosine distance differences so PCA has
meaningful variance to work with.

Size driven by salience (wide variation).
Height driven by relevance to article.
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np


def force_directed_repulsion(candidates, iterations=700, min_distance=0.255, repulsion=0.005, damping=0.86, center_pull=0.00015):
    """
    Strong force repulsion. Creates water channels.
    min_distance is large (0.15) so even small topics get pushed apart.
    repulsion is strong (0.003) so gaps form quickly.
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

                # Both topics' sizes affect minimum distance
                min_d = min_distance + (pts[i]["size"] + pts[j]["size"]) * 2.0

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
    3. StandardScaler (from notebook — amplifies small differences)
    4. PCA to 2D
    5. Size from salience, height from relevance
    6. Strong force repulsion
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

    # Embed entities text for relevance scoring
    entities_emb = embed_texts([entities_text])[0] if entities_text.strip() else None
    article_emb = embed_texts([article_text])[0]

    # ── StandardScaler + PCA (from notebook Cell 31) ─────────────
    # This is the KEY fix. StandardScaler normalizes each column of
    # the distance matrix to mean=0, std=1. This amplifies small
    # cosine distance differences so PCA has variance to work with.
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(dist_matrix)

    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components, random_state=1)
    pca.fit(scaled_matrix)
    coords = pca.components_

    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    # ── Compute size and height per candidate ────────────────────
    debug_info = []

    for i, candidate in enumerate(candidates):
        sim_article = float(cosine_similarity(embeddings[i], article_emb))
        sim_entities = float(cosine_similarity(embeddings[i], entities_emb)) if entities_emb is not None else 0.0
        tfidf_rel = candidate.get("tfidf_relevance", 0.0)

        # SIZE = salience score (from NER)
        # This has WIDE variation (0.001 to 0.35) like Google NLP
        # Drives the x-y footprint of the half-ellipse
        salience = candidate.get("base_score", 0.1)
        # For wikipedia_search candidates, use embedding similarity as proxy
        if candidate.get("source") == "wikipedia_search":
            salience = candidate.get("similarity", 0.1) * 0.5

        # HEIGHT = relevance to the article content
        # Drives the z-axis of the half-ellipse
        raw_height = max(sim_entities, sim_article * 0.6, tfidf_rel)

        candidate["_salience"] = salience
        candidate["_raw_height"] = raw_height
        candidate["_sim_article"] = sim_article
        candidate["_sim_entities"] = sim_entities
        candidate["_tfidf"] = tfidf_rel

    # ── Normalize salience to size range ─────────────────────────
    # Target: 0.02 (tiny islet) to 0.09 (dominant island) before radius_mult
    # Use salience directly — it already has wide variation from NER
    saliences = np.array([c["_salience"] for c in candidates])
    sal_min, sal_max = saliences.min(), saliences.max()
    if sal_max > sal_min:
        sal_norm = (saliences - sal_min) / (sal_max - sal_min)
    else:
        sal_norm = np.ones_like(saliences) * 0.5

    # ── Normalize height ─────────────────────────────────────────
    raw_heights = np.array([c["_raw_height"] for c in candidates])
    h_min, h_max = raw_heights.min(), raw_heights.max()
    if h_max > h_min:
        h_norm = (raw_heights - h_min) / (h_max - h_min)
    else:
        h_norm = np.ones_like(raw_heights) * 0.5

    # Apply mild power curve to height (less aggressive than before)
    h_norm = np.power(h_norm, 1.5)

    # ── Assign values ────────────────────────────────────────────
    for i, candidate in enumerate(candidates):
        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier

        # Size from salience (wide range: 4-5x ratio between max and min)
        candidate["size"] = float(0.02 + sal_norm[i] * 0.07) * radius_multiplier

        # Height from relevance (0 to 1, frontend normalizes to v3 range)
        candidate["height"] = float(h_norm[i])

        debug_info.append({
            "title": candidate["title"],
            "source": candidate.get("source", ""),
            "salience": round(candidate["_salience"], 4),
            "sim_article": round(candidate["_sim_article"], 4),
            "sim_entities": round(candidate["_sim_entities"], 4),
            "tfidf": round(candidate["_tfidf"], 4),
            "raw_height": round(candidate["_raw_height"], 4),
            "final_height": round(candidate["height"], 4),
            "final_size": round(candidate["size"], 4),
            "x_pca": round(candidate["x"], 4),
            "y_pca": round(candidate["y"], 4),
        })

    # ── Force repulsion ──────────────────────────────────────────
    candidates = force_directed_repulsion(
        candidates,
        iterations=600,
        min_distance=0.15,
        repulsion=0.003,
        damping=0.88,
        center_pull=0.0002,
    )

    # Update debug with post-repulsion positions
    for i, d in enumerate(debug_info):
        d["x_final"] = round(candidates[i]["x"], 4)
        d["y_final"] = round(candidates[i]["y"], 4)

    # Distance matrix for debug
    dm_list = []
    topic_labels = [c["title"] for c in candidates]
    for i in range(len(candidates)):
        row = {}
        for j in range(len(candidates)):
            row[topic_labels[j]] = round(float(dist_matrix[i][j]), 4)
        dm_list.append({"topic": topic_labels[i], "distances": row})

    # Cleanup
    for c in candidates:
        for k in ["_salience", "_raw_height", "_sim_article", "_sim_entities", "_tfidf"]:
            c.pop(k, None)

    return {
        "candidates": candidates,
        "distance_matrix": dm_list,
        "debug": debug_info,
    }
