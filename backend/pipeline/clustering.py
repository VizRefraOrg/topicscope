"""
Topic processing: passes ALL candidates through.
Every entity appears with its own x, y, size, height.
No elimination, no splitting.
The original system plotted every entity on the 2D scatter.
"""

import numpy as np


def process_topics(candidates: list[dict]) -> list[dict]:
    """
    Pass ALL candidates through with normalized heights.
    No splitting into islands/shore markers.
    Every entity gets its own position on all views.
    """
    if not candidates:
        return []

    # Normalize heights to 0-1
    max_h = max(c["height"] for c in candidates) if candidates else 1
    if max_h > 0:
        for c in candidates:
            c["height"] = c["height"] / max_h

    # Sort by size descending (largest first)
    candidates.sort(key=lambda c: c["size"], reverse=True)

    return candidates
