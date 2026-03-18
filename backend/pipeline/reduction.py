"""
PCA + t-SNE reduction, HDBSCAN clustering, heightmap generation.

- size = salience / (salience[0] / entity_text_sim[0])
- height = entity_text_sim (TF-IDF cosine sim to full text)
- Cosine distance via Azure OpenAI embeddings
- PCA(50) → t-SNE(2D) for layout
- sklearn HDBSCAN clustering with adaptive parameters
- Gaussian heightmap terrain generation
"""

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from scipy.ndimage import gaussian_filter
from backend.pipeline.embeddings import embed_texts, cosine_distance_matrix, cosine_similarity
import numpy as np


def reduce_embeddings(embeddings, n_entities):
    """PCA to 50 dims then t-SNE to 2D. Handles small-N edge cases."""
    N = n_entities

    if N == 1:
        return np.array([[0.0, 0.0]])

    if N == 2:
        pca = PCA(n_components=1, random_state=42)
        coords_1d = pca.fit_transform(embeddings)
        return np.hstack([coords_1d, np.zeros((N, 1))])

    if N < 5:
        n_comp = min(2, N - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(embeddings)
        if n_comp == 1:
            coords = np.hstack([coords, np.zeros((N, 1))])
        return coords

    # Stage 1: PCA to 50 dims (or N-1 if fewer)
    n_pca = min(50, N - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    reduced = pca.fit_transform(embeddings)

    # Stage 2: t-SNE to 2D
    perplexity = max(5, min(30, N // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        metric='cosine',
        random_state=42,
        n_iter=1000,
    )
    coords_2d = tsne.fit_transform(reduced)
    return coords_2d


def cluster_entities(coords_2d, n_entities):
    """HDBSCAN clustering with adaptive params. Assigns noise to nearest cluster."""
    N = n_entities

    if N < 4:
        return np.ones(N, dtype=int)

    # Adaptive parameters
    if N <= 20:
        min_cluster_size = 2
        min_samples = 1
    elif N <= 50:
        min_cluster_size = 3
        min_samples = 2
    else:
        min_cluster_size = 4
        min_samples = 2

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(coords_2d)

    # If all noise, fall back to AgglomerativeClustering
    if np.all(labels == -1):
        n_clusters = max(2, min(8, N // 8))
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(coords_2d)
        return labels + 1  # 1-based

    # Assign noise points (-1) to nearest non-noise neighbor
    noise_mask = labels == -1
    if np.any(noise_mask):
        non_noise_indices = np.where(~noise_mask)[0]
        for i in np.where(noise_mask)[0]:
            dists = np.linalg.norm(coords_2d[non_noise_indices] - coords_2d[i], axis=1)
            nearest = non_noise_indices[np.argmin(dists)]
            labels[i] = labels[nearest]

    # Re-index to 1-based contiguous labels
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels, start=1)}
    labels = np.array([label_map[l] for l in labels])

    return labels


def generate_heightmap(coords_2d, saliences, heights, grid_size=200,
                       sigma_pct=0.045, sea_level_percentile=35):
    """Generate a Gaussian terrain heightmap from entity positions and weights."""
    N = len(coords_2d)

    # Edge case: increase sigma for small N
    if N < 10:
        sigma_pct = 0.07

    margin = int(grid_size * 0.12)

    # Normalize coords to [margin, grid_size - margin]
    coords = np.array(coords_2d, dtype=float)
    for dim in range(2):
        cmin, cmax = coords[:, dim].min(), coords[:, dim].max()
        span = cmax - cmin
        if span < 1e-9:
            coords[:, dim] = grid_size / 2.0
        else:
            coords[:, dim] = margin + (coords[:, dim] - cmin) / span * (grid_size - 2 * margin)

    # Weights = saliences * heights, normalized to [0, 1]
    weights = np.array(saliences) * np.array(heights)
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-9:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = np.ones(N)

    # Place weighted points on grid (SUM accumulation)
    grid = np.zeros((grid_size, grid_size), dtype=float)
    entity_grid_positions = []
    for i in range(N):
        gx = int(np.clip(coords[i, 0], 0, grid_size - 1))
        gy = int(np.clip(coords[i, 1], 0, grid_size - 1))
        grid[gy, gx] += weights[i]
        entity_grid_positions.append({"gx": gx, "gy": gy})

    # Broad Gaussian pass
    sigma = sigma_pct * grid_size
    broad = gaussian_filter(grid, sigma=sigma)

    # Peaked Gaussian pass (40% blend)
    peak_sigma = sigma * 0.4
    peaked = gaussian_filter(grid, sigma=peak_sigma)

    # Blend: 60% broad + 40% peaked
    blended = 0.6 * broad + 0.4 * peaked

    # Sea level: percentile of non-zero values, subtract and re-normalize
    nonzero_vals = blended[blended > 0.001]
    if len(nonzero_vals) > 0:
        sea_level = np.percentile(nonzero_vals, sea_level_percentile)
        blended = blended - sea_level
        blended = np.clip(blended, 0, None)
        bmax = blended.max()
        if bmax > 1e-9:
            blended = blended / bmax

    bounds = {
        "x_min": float(margin),
        "x_max": float(grid_size - margin),
        "y_min": float(margin),
        "y_max": float(grid_size - margin),
    }

    return {
        "heightmap": [[round(v, 4) for v in row] for row in blended.tolist()],
        "grid_size": grid_size,
        "sea_level": 0.0,
        "bounds": bounds,
        "entity_grid_positions": entity_grid_positions,
    }


def compute_distance_and_reduce(
    candidates: list[dict],
    article_text: str,
    entities: list[dict] = None,
    entities_text: str = "",
) -> dict:
    if len(candidates) < 2:
        for c in candidates:
            c["x"] = c["y"] = 0.0
            c["size"] = 0.05
            c["height"] = 0.5
            c["cluster"] = 1
        return {"candidates": candidates, "distance_matrix": [], "heightmap": {}, "debug": []}

    # Build entity lookup for tag/salience
    entity_lookup = {}
    if entities:
        for e in entities:
            entity_lookup[e["name"].lower()] = e

    titles = [c["title"] for c in candidates]
    embeddings = embed_texts(titles)
    dist_matrix = cosine_distance_matrix(embeddings)

    # ── TF-IDF similarity to full text (height) ──────────────────
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

    sizes = np.clip(sizes, 0.005, 1.0)

    # ── PCA + t-SNE reduction ────────────────────────────────────
    N = len(candidates)
    coords_2d = reduce_embeddings(embeddings, N)

    # ── HDBSCAN clustering ───────────────────────────────────────
    cluster_labels = cluster_entities(coords_2d, N)

    # ── Heightmap generation ─────────────────────────────────────
    heightmap_data = generate_heightmap(coords_2d, saliences, entity_text_sim)

    # ── Assign values ────────────────────────────────────────────
    debug_info = []

    for i, candidate in enumerate(candidates):
        candidate["x"] = float(coords_2d[i][0])
        candidate["y"] = float(coords_2d[i][1])
        candidate["size"] = float(sizes[i])
        candidate["height"] = float(entity_text_sim[i])
        candidate["tag"] = tags[i]
        candidate["cluster"] = int(cluster_labels[i])
        candidate["salience"] = float(saliences[i])
        # Embed grid position in candidate so it survives sorting in clustering.py
        if heightmap_data.get("entity_grid_positions"):
            candidate["grid_gx"] = heightmap_data["entity_grid_positions"][i]["gx"]
            candidate["grid_gy"] = heightmap_data["entity_grid_positions"][i]["gy"]

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
            "x_final": round(candidate["x"], 4),
            "y_final": round(candidate["y"], 4),
        })

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
        "heightmap": heightmap_data,
        "debug": debug_info,
    }
