"""
Topic processing: splits entities into islands (large) and shore markers (small).
No more elimination - all discovered entities appear in the visualization.
Small entities are assigned to the nearest large island as shore markers.
"""

import numpy as np


def process_topics(candidates: list[dict]) -> list[dict]:
    """
    Process all candidates into two categories:
    - Islands: large/important topics become full 3D islands
    - Shore markers: smaller topics placed on edges of nearest island

    ALL entities appear in the final output - nothing is eliminated.
    """
    if not candidates:
        return []

    # Sort by size descending
    sorted_cands = sorted(candidates, key=lambda c: c["size"], reverse=True)

    # Determine island threshold: top ~40% by size become islands
    # Minimum 3 islands, maximum 15
    n_islands = max(3, min(15, int(len(sorted_cands) * 0.4)))

    islands = sorted_cands[:n_islands]
    shore_candidates = sorted_cands[n_islands:]

    # Normalize island heights to 0-1
    max_h = max(c["height"] for c in islands) if islands else 1
    if max_h > 0:
        for c in islands:
            c["height"] = c["height"] / max_h

    # Mark islands and init shore_markers list
    for c in islands:
        c["role"] = "island"
        c["shore_markers"] = []

    # Assign each shore candidate to nearest island
    if islands and shore_candidates:
        island_positions = np.array([[c["x"], c["y"]] for c in islands])

        for sc in shore_candidates:
            dists = np.sqrt((island_positions[:, 0] - sc["x"])**2 + (island_positions[:, 1] - sc["y"])**2)
            nearest_idx = int(np.argmin(dists))

            islands[nearest_idx]["shore_markers"].append({
                "label": sc["title"],
                "similarity": sc.get("similarity", 0),
                "source": sc.get("source", ""),
                "wikipedia_url": sc.get("wikipedia_url", ""),
                "size": sc["size"],
            })

    islands.sort(key=lambda c: c["height"], reverse=True)
    return islands
