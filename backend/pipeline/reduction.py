"""
StandardScaler + PCA, sqrt size scaling, K-means clustering.

Key changes from original analysis:
- Size computed as sqrt(salience) * 100 (ready to render, no frontend normalization)
- Cluster IDs via K-means on positions
- Entity type (tag) passed through for color coding
- StandardScaler before PCA (from notebook)
- Strong force repulsion
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np


def force_directed_repulsion(candidates, iterations=700, min_distance=0.255, repulsion=0.005, damping=0.86, center_pull=0.00015):
    pts = [{"x": c["x"], "y": c["y"], "vx": 0, "vy": 0, "size": c.get("size", 30)} for c in candidates]
    for _ in range(iterations):
        for p in pts:
            p["fx"] = p["fy"] = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i]["x"] - pts[j]["x"]
                dy = pts[i]["y"] - pts[j]["y"]
                dist = max(np.sqrt(dx * dx + dy * dy), 0.001)
                # Scale min_d by actual sizes (now in 0-100 range, need to normalize)
                sz_factor = (pts[i]["size"] + pts[j]["size"]) / 100 * 2.0
                min_d = min_distance + sz_factor
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
    entities: list[dict] = None,
    entities_text: str = "",
    spacing_multiplier: float = 3.0,
) -> dict:
    if len(candidates) < 2:
        for c in candidates:
            c["x"] = 0.0
            c["y"] = 0.0
            c["size"] = 30
            c["height"] = 0.5
            c["cluster"] = 1
        return {"candidates": candidates, "distance_matrix": [], "debug": []}

    # Build entity lookup for tag/salience passthrough
    entity_lookup = {}
    if entities:
        for e in entities:
            entity_lookup[e["name"].lower()] = e

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    article_emb = embed_texts([article_text])[0]
    entities_emb = embed_texts([entities_text])[0] if entities_text.strip() else None

    # ── StandardScaler + PCA ─────────────────────────────────────
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(dist_matrix)
    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components, random_state=1)
    pca.fit(scaled_matrix)
    coords = pca.components_
    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    # ── K-means clustering on positions ──────────────────────────
    positions_for_cluster = np.array([[coords[0][i], coords[1][i]] for i in range(len(candidates))])
    n_clusters = min(5, max(2, len(candidates) // 4))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(positions_for_cluster)

    # ── Compute scores per candidate ─────────────────────────────
    debug_info = []

    for i, candidate in enumerate(candidates):
        sim_article = float(cosine_similarity(embeddings[i], article_emb))
        sim_entities = float(cosine_similarity(embeddings[i], entities_emb)) if entities_emb is not None else 0.0
        tfidf_rel = candidate.get("tfidf_relevance", 0.0)

        # Look up entity salience and tag
        title_lower = candidate["title"].lower()
        entity_match = entity_lookup.get(title_lower, {})
        salience = entity_match.get("salience", candidate.get("base_score", 0.1))
        tag = entity_match.get("type", "MISC")

        # ── SIZE = sqrt(salience) × 100 ─────────────────────────
        # sqrt scaling: if salience doubles, AREA doubles (not radius)
        # Range: sqrt(0.01)*100=10 to sqrt(0.95)*100=97
        candidate["size"] = float(np.sqrt(max(salience, 0.005)) * 100)

        # ── HEIGHT = relevance to article (0 to 1) ──────────────
        raw_height = max(sim_entities, sim_article * 0.6, tfidf_rel)
        candidate["height"] = float(raw_height)

        # ── TAG = entity type for color coding ───────────────────
        candidate["tag"] = tag

        # ── CLUSTER = K-means group ID ───────────────────────────
        candidate["cluster"] = int(cluster_labels[i]) + 1  # 1-indexed

        # ── POSITION ─────────────────────────────────────────────
        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier

        # ── SALIENCE passthrough ─────────────────────────────────
        candidate["salience"] = float(salience)

        debug_info.append({
            "title": candidate["title"],
            "source": candidate.get("source", ""),
            "tag": tag,
            "cluster": candidate["cluster"],
            "salience": round(salience, 4),
            "size": round(candidate["size"], 1),
            "sim_article": round(sim_article, 4),
            "sim_entities": round(sim_entities, 4),
            "tfidf": round(tfidf_rel, 4),
            "raw_height": round(raw_height, 4),
            "final_height": round(candidate["height"], 4),
            "x": round(candidate["x"], 4),
            "y": round(candidate["y"], 4),
        })

    # ── Force repulsion ──────────────────────────────────────────
    candidates = force_directed_repulsion(candidates)

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

    return {
        "candidates": candidates,
        "distance_matrix": dm_list,
        "debug": debug_info,
    }
