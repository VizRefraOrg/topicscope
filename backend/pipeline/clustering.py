"""
Topic processing: splits into islands + shore markers.
All entities appear. Passes through tag and cluster.
"""

import numpy as np


def process_topics(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c["size"], reverse=True)

    # Top 40% by size become islands (min 3, max 15)
    n_islands = max(3, min(15, int(len(sorted_cands) * 0.4)))
    islands = sorted_cands[:n_islands]
    shore_candidates = sorted_cands[n_islands:]

    # Normalize island heights to 0-1
    max_h = max(c["height"] for c in islands) if islands else 1
    if max_h > 0:
        for c in islands:
            c["height"] = c["height"] / max_h

    for c in islands:
        c["role"] = "island"
        c["shore_markers"] = []

    # Assign shore candidates to nearest island
    if islands and shore_candidates:
        island_positions = np.array([[c["x"], c["y"]] for c in islands])
        for sc in shore_candidates:
            dists = np.sqrt((island_positions[:, 0] - sc["x"])**2 + (island_positions[:, 1] - sc["y"])**2)
            nearest_idx = int(np.argmin(dists))
            islands[nearest_idx]["shore_markers"].append({
                "label": sc["title"],
                "tag": sc.get("tag", "MISC"),
                "similarity": sc.get("similarity", 0),
                "source": sc.get("source", ""),
                "wikipedia_url": sc.get("wikipedia_url", ""),
                "size": sc["size"],
            })

    islands.sort(key=lambda c: c["height"], reverse=True)
    return islands
