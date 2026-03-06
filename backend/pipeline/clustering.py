"""
Circle overlap elimination and K-means integration.
Ported directly from main_2020.py logic.
"""

from sklearn.cluster import KMeans
import numpy as np


def circle_overlap(x1, y1, x2, y2, r1, r2, overlap_factor=1.5) -> bool:
    """Check if two circles overlap (same as original)."""
    dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    rad_sum_sq = (r1 + r2) ** 2
    return dist_sq * overlap_factor < rad_sum_sq


def eliminate_overlaps(candidates: list[dict]) -> list[dict]:
    """
    Elimination: larger circles absorb overlapping smaller circles.
    Same logic as main_2020.py circle elimination.
    """
    # Sort by size descending (largest first)
    sorted_candidates = sorted(candidates, key=lambda c: c["size"], reverse=True)

    for c in sorted_candidates:
        c["_deleted"] = False

    for i in range(len(sorted_candidates) - 1):
        if sorted_candidates[i]["_deleted"]:
            continue
        for j in range(i + 1, len(sorted_candidates)):
            if sorted_candidates[j]["_deleted"]:
                continue
            ci = sorted_candidates[i]
            cj = sorted_candidates[j]
            if circle_overlap(ci["x"], ci["y"], cj["x"], cj["y"], ci["size"], cj["size"]):
                if ci["size"] >= cj["size"]:
                    cj["_deleted"] = True
                else:
                    ci["_deleted"] = True

    surviving = [c for c in sorted_candidates if not c["_deleted"]]

    # Clean up
    for c in sorted_candidates:
        if "_deleted" in c:
            del c["_deleted"]

    return surviving


def integrate_clusters(
    candidates: list[dict],
    size_threshold: float = 0.03,
    n_clusters: int = 8,
    top_n_per_cluster: int = 3,
) -> list[dict]:
    """
    Integration: small scattered entities are clustered via K-means
    and represented by top entities per cluster.
    Same logic as main_2020.py integration step.
    """
    # Split into large (keep as-is) and small (to be clustered)
    large = [c for c in candidates if c["size"] >= size_threshold]
    small = [c for c in candidates if c["size"] < size_threshold]

    if len(small) < n_clusters:
        # Not enough small items to cluster meaningfully
        return large + small

    # K-means on small items' positions
    positions = np.array([[c["x"], c["y"]] for c in small])
    actual_clusters = min(n_clusters, len(small))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init=10)
    kmeans.fit(positions)

    # From each cluster, take top_n by height
    cluster_representatives = []
    for cluster_num in range(actual_clusters):
        cluster_items = [c for c, label in zip(small, kmeans.labels_) if label == cluster_num]
        cluster_items.sort(key=lambda c: c["height"], reverse=True)
        top_items = cluster_items[:top_n_per_cluster]
        cluster_representatives.extend(top_items)

    return large + cluster_representatives


def process_topics(candidates: list[dict]) -> list[dict]:
    """
    Full elimination + integration pipeline.
    Returns final list of topics ready for visualization.
    """
    # Step 1: Eliminate overlapping circles
    surviving = eliminate_overlaps(candidates)

    # Step 2: Integrate small clusters
    final = integrate_clusters(surviving)

    # Normalize heights to 0-1 range
    if final:
        max_h = max(c["height"] for c in final)
        if max_h > 0:
            for c in final:
                c["height"] = c["height"] / max_h

    # Sort by height descending for consistent output
    final.sort(key=lambda c: c["height"], reverse=True)

    return final
