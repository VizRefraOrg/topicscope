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


def generate_heightmap(coords_2d, saliences, heights, grid_size=200):
    """Generate a Gaussian terrain heightmap from entity positions and weights."""
    from scipy.spatial.distance import cdist
    N = len(coords_2d)

    margin = int(grid_size * 0.10)

    # Normalize coords to [margin, grid_size - margin]
    coords = np.array(coords_2d, dtype=float)
    for dim in range(2):
        cmin, cmax = coords[:, dim].min(), coords[:, dim].max()
        span = cmax - cmin
        if span < 1e-9:
            coords[:, dim] = grid_size / 2.0
        else:
            coords[:, dim] = margin + (coords[:, dim] - cmin) / span * (grid_size - 2 * margin)

    # FIX 2: Log weight compression + high floor (0.40)
    # Prevents one entity from creating a 40:1 spike
    raw_weights = np.array(saliences) * np.array(heights)
    raw_weights = np.clip(raw_weights, 1e-6, None)
    weights = np.log1p(raw_weights)  # log(1+x) compresses more than sqrt
    w_max = weights.max()
    if w_max > 1e-9:
        weights = weights / w_max
    weights = np.clip(weights, 0.40, 1.0)

    # Place weighted points on grid (SUM accumulation)
    grid = np.zeros((grid_size, grid_size), dtype=float)
    entity_grid_positions = []
    for i in range(N):
        gx = int(np.clip(coords[i, 0], 0, grid_size - 1))
        gy = int(np.clip(coords[i, 1], 0, grid_size - 1))
        grid[gy, gx] += weights[i]
        entity_grid_positions.append({"gx": gx, "gy": gy})

    # Adaptive sigma — tighter cap to create distinct islands
    if N >= 2:
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        nearest_dists = dists.min(axis=1)
        median_nn = np.median(nearest_dists)
        sigma = max(median_nn * 0.9, grid_size * 0.04)
        sigma = min(sigma, grid_size * 0.12)
    else:
        sigma = grid_size * 0.08

    # Peaked-dominant blend (40/60) — more elevation contrast between peaks
    broad = gaussian_filter(grid, sigma=sigma)
    peak_sigma = max(sigma * 0.35, 4.0)
    peaked = gaussian_filter(grid, sigma=peak_sigma)
    blended = 0.40 * broad + 0.60 * peaked

    # Percentile normalization: use 98th percentile as ceiling
    # Prevents one spike from compressing all other terrain into green zone
    land_vals = blended[blended > 1e-9]
    if len(land_vals) > 10:
        p98 = np.percentile(land_vals, 98)
        if p98 > 1e-9:
            blended = blended / p98
            blended = np.clip(blended, 0.0, 1.0)
    else:
        bmax = blended.max()
        if bmax > 1e-9:
            blended = blended / bmax

    # Island mask: tighter radius (1.6x) to carve ocean between clusters
    island_radius = sigma * 1.6
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for i in range(N):
        gx = int(np.clip(coords[i, 0], 0, grid_size - 1))
        gy = int(np.clip(coords[i, 1], 0, grid_size - 1))
        dist_sq = (xx - gx) ** 2 + (yy - gy) ** 2
        mask |= (dist_sq < island_radius ** 2)
    blended[~mask] = 0.0

    # Gamma 0.55: stronger lift to push mid-values into tan/brown color bands
    blended = np.power(blended, 0.55)

    # Higher sea cutoff carves sharper shores and wider ocean channels
    sea_cutoff = 0.15
    blended = np.where(blended < sea_cutoff, 0.0, (blended - sea_cutoff) / (1.0 - sea_cutoff))

    # Histogram equalization on land: spreads values across full 0-1 range
    # so ALL color stops (green -> olive -> tan -> brown -> white) are used
    land_mask_final = blended > 0.001
    land_vals_final = blended[land_mask_final]
    if len(land_vals_final) > 50:
        sorted_vals = np.sort(land_vals_final)
        ranks = np.searchsorted(sorted_vals, blended[land_mask_final], side='right')
        equalized = ranks.astype(float) / len(sorted_vals)
        # Blend 55% equalized + 45% original to keep natural terrain shape
        blended[land_mask_final] = 0.55 * equalized + 0.45 * blended[land_mask_final]

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


def compute_entity_grid_positions(entities, candidates, heightmap_data, grid_size=200):
    """Compute grid positions for ALL NER entities, including those not in candidates.

    Entities matching a candidate get that candidate's grid position.
    Non-matching entities are placed at the shore of the nearest matching entity's island.
    """
    if not entities or not heightmap_data.get("entity_grid_positions"):
        return []

    margin = int(grid_size * 0.10)
    candidate_titles_lower = {c["title"].lower(): i for i, c in enumerate(candidates)}

    entity_positions = []
    for ent in entities:
        name_lower = ent["name"].lower()

        if name_lower in candidate_titles_lower:
            idx = candidate_titles_lower[name_lower]
            gpos = heightmap_data["entity_grid_positions"][idx]
            entity_positions.append({
                "name": ent["name"],
                "type": ent.get("type", "MISC"),
                "confidence": ent.get("confidence", 0.0),
                "salience": ent.get("salience", 0.0),
                "grid_gx": gpos["gx"],
                "grid_gy": gpos["gy"],
                "is_topic": True,
            })
        else:
            # Find nearest candidate by name overlap; distribute evenly if no match
            best_idx = hash(name_lower) % len(candidates)  # default: distribute by hash
            best_score = 0
            for title_lower, idx in candidate_titles_lower.items():
                score = 0
                if name_lower in title_lower or title_lower in name_lower:
                    score = 2
                else:
                    words_e = set(name_lower.split())
                    words_c = set(title_lower.split())
                    overlap = words_e & words_c
                    score = len(overlap) / max(len(words_e), 1)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            gpos = heightmap_data["entity_grid_positions"][best_idx]
            base_gx, base_gy = gpos["gx"], gpos["gy"]

            # Search radially outward for a shore position (low but non-zero elevation)
            hmap = np.array(heightmap_data["heightmap"])
            best_shore_gx, best_shore_gy = base_gx, base_gy
            best_shore_h = 999
            for angle_step in range(8):
                angle = angle_step * (np.pi / 4)
                for r in range(5, 40, 3):
                    sx = int(base_gx + r * np.cos(angle))
                    sy = int(base_gy + r * np.sin(angle))
                    if 0 <= sx < grid_size and 0 <= sy < grid_size:
                        sh = hmap[sy, sx]
                        if 0.001 < sh < 0.35 and sh < best_shore_h:
                            best_shore_h = sh
                            best_shore_gx = sx
                            best_shore_gy = sy

            # Fallback: offset slightly from parent entity
            if best_shore_h == 999:
                rng = np.random.RandomState(hash(name_lower) % 2**31)
                offset_x = rng.randint(-10, 11)
                offset_y = rng.randint(-10, 11)
                best_shore_gx = int(np.clip(base_gx + offset_x, margin, grid_size - margin))
                best_shore_gy = int(np.clip(base_gy + offset_y, margin, grid_size - margin))

            entity_positions.append({
                "name": ent["name"],
                "type": ent.get("type", "MISC"),
                "confidence": ent.get("confidence", 0.0),
                "salience": ent.get("salience", 0.0),
                "grid_gx": best_shore_gx,
                "grid_gy": best_shore_gy,
                "is_topic": False,
            })

    return entity_positions


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

    # Compute grid positions for ALL NER entities (for 3D label placement)
    all_entity_positions = compute_entity_grid_positions(
        entities or [], candidates, heightmap_data
    )

    return {
        "candidates": candidates,
        "distance_matrix": dm_list,
        "heightmap": heightmap_data,
        "debug": debug_info,
        "all_entity_positions": all_entity_positions,
    }
