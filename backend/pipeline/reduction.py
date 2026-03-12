"""
StandardScaler + PCA, original size formula, HDBSCAN clustering.

Matches the original models.py create_circles() logic:
- size = salience / (salience[0] / entity_text_sim[0])  
- height = entity_text_sim (TF-IDF cosine sim to full text)
- WMD replaced by cosine distance (Azure OpenAI embeddings)
- StandardScaler + PCA (same as original)
- HDBSCAN clustering (same as original, min_cluster_size=3, min_samples=1)
- Force repulsion (not in original, but needed for Azure's narrow distance range)
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    from sklearn.cluster import KMeans
    HAS_HDBSCAN = False


def force_directed_repulsion(candidates, iterations=800, min_distance=0.35, repulsion=0.008, damping=0.85, center_pull=0.0001):
    pts = [{"x": c["x"], "y": c["y"], "vx": 0, "vy": 0, "size": c.get("size", 0.05)} for c in candidates]
    for _ in range(iterations):
        for p in pts:
            p["fx"] = p["fy"] = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i]["x"] - pts[j]["x"]
                dy = pts[i]["y"] - pts[j]["y"]
                dist = max(np.sqrt(dx * dx + dy * dy), 0.001)
                # Size factor: sqrt of combined sizes (matches D3's scaleSqrt)
                # Sizes are 0.05-0.8 range from original formula
                sz_factor = np.sqrt(pts[i]["size"] + pts[j]["size"]) * 0.6
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
            c["x"] = c["y"] = 0.0
            c["size"] = 0.05
            c["height"] = 0.5
            c["cluster"] = 1
        return {"candidates": candidates, "distance_matrix": [], "debug": []}

    # Build entity lookup for tag/salience
    entity_lookup = {}
    if entities:
        for e in entities:
            entity_lookup[e["name"].lower()] = e

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    # ── TF-IDF similarity to full text (original's height) ───────
    # Matches: height = entity_text_sim from models.py
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        clean_titles = [t.lower() for t in titles]
        tfidf_model = vectorizer.fit(clean_titles)
        entity_vectors = tfidf_model.transform(clean_titles)
        text_vector = tfidf_model.transform([article_text.lower()])
        entity_text_sim = sklearn_cosine_sim(entity_vectors, text_vector).ravel()
    except Exception:
        entity_text_sim = np.ones(len(candidates)) * 0.5

    # ── Size formula from original models.py ─────────────────────
    # size = salience / (salience[0] / entity_text_sim[0])
    # This rescales salience by the ratio of top entity's TF-IDF sim
    saliences = np.array([
        entity_lookup.get(c["title"].lower(), {}).get("salience", c.get("base_score", 0.1))
        for c in candidates
    ])
    tags = [
        entity_lookup.get(c["title"].lower(), {}).get("type", "MISC")
        for c in candidates
    ]

    if saliences[0] > 0 and entity_text_sim[0] > 0:
        ratio = saliences[0] / entity_text_sim[0]
        sizes = saliences / ratio
    else:
        sizes = saliences

    # Clamp sizes to reasonable range
    sizes = np.clip(sizes, 0.005, 1.0)

    # ── StandardScaler + PCA (same as original) ──────────────────
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(dist_matrix)

    n_components = min(2, len(candidates) - 1)
    pca = PCA(n_components=n_components, random_state=1)
    pca.fit(scaled_matrix)
    coords = pca.components_
    if n_components == 1:
        coords = np.vstack([coords, np.zeros_like(coords[0])])

    # ── HDBSCAN clustering (same as original) ────────────────────
    # Original: hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
    if HAS_HDBSCAN and len(candidates) >= 4:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
        cluster_labels = clusterer.fit_predict(scaled_matrix)
        # HDBSCAN uses -1 for noise; remap to positive
        cluster_labels = [max(0, c) + 1 for c in cluster_labels]
    else:
        # Fallback: K-means
        n_clusters = min(5, max(2, len(candidates) // 4))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = (kmeans.fit_predict(scaled_matrix) + 1).tolist()

    # ── Assign values ────────────────────────────────────────────
    debug_info = []

    for i, candidate in enumerate(candidates):
        candidate["x"] = float(coords[0][i]) * spacing_multiplier
        candidate["y"] = float(coords[1][i]) * spacing_multiplier
        candidate["size"] = float(sizes[i])
        candidate["height"] = float(entity_text_sim[i])
        candidate["tag"] = tags[i]
        candidate["cluster"] = int(cluster_labels[i])
        candidate["salience"] = float(saliences[i])

        debug_info.append({
            "title": candidate["title"],
            "source": candidate.get("source", ""),
            "tag": tags[i],
            "cluster": int(cluster_labels[i]),
            "salience": round(float(saliences[i]), 4),
            "size": round(float(sizes[i]), 4),
            "tfidf_sim": round(float(entity_text_sim[i]), 4),
            "height": round(float(entity_text_sim[i]), 4),
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
