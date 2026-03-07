"""
Topic lookup: discovers candidate topics from Wikipedia.
More aggressive search strategy: uses entity combinations to find
rich compound topic titles like "Russian espionage", "nerve agent attack",
"military intelligence" instead of just raw entity names.
"""

import httpx
import asyncio
from backend.pipeline.embeddings import embed_texts, cosine_similarity
import numpy as np


async def search_wikipedia(query: str, limit: int = 10) -> list[dict]:
    """Search Wikipedia for articles related to the query."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "srprop": "snippet",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            data = response.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            results.append({
                "title": title,
                "snippet": item.get("snippet", ""),
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            })
        return results
    except Exception:
        return []


async def find_candidate_topics(
    article_text: str,
    entities: list[dict],
    entity_links: list[dict],
    max_candidates: int = 60,
) -> list[dict]:
    """
    Find candidate topics from multiple sources.
    More aggressive search strategy with entity combinations.
    """
    candidates = {}

    # Source 1: Entity-linked Wikipedia articles
    for link in entity_links:
        title = link["name"]
        key = title.lower()
        if key not in candidates:
            candidates[key] = {
                "title": title,
                "source": "entity_link",
                "wikipedia_url": link.get("wikipedia_url", ""),
                "base_score": link["confidence"],
            }

    # Source 2: Wikipedia search with individual top entities
    top_entities = [e["name"] for e in entities[:10]]
    search_tasks = []

    for entity_name in top_entities:
        search_tasks.append(search_wikipedia(entity_name, limit=8))

    # Source 3: Wikipedia search with COMBINATIONS of entities
    # This is what finds compound topics like "Russian espionage",
    # "nerve agent attack", "military intelligence"
    if len(top_entities) >= 2:
        for i in range(min(5, len(top_entities))):
            for j in range(i + 1, min(6, len(top_entities))):
                combo = f"{top_entities[i]} {top_entities[j]}"
                search_tasks.append(search_wikipedia(combo, limit=5))

    # Source 4: Search with 3-entity combinations for broader coverage
    if len(top_entities) >= 3:
        for i in range(min(3, len(top_entities))):
            combo3 = f"{top_entities[i]} {top_entities[min(i+1, len(top_entities)-1)]} {top_entities[min(i+2, len(top_entities)-1)]}"
            search_tasks.append(search_wikipedia(combo3, limit=5))

    # Run all searches concurrently
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result_list in search_results:
        if isinstance(result_list, Exception):
            continue
        for item in result_list:
            key = item["title"].lower()
            if key not in candidates:
                candidates[key] = {
                    "title": item["title"],
                    "source": "wikipedia_search",
                    "wikipedia_url": item.get("url", ""),
                    "base_score": 0.5,
                }

    # Source 5: Entities themselves as candidates
    for entity in entities[:15]:
        key = entity["name"].lower()
        if key not in candidates:
            candidates[key] = {
                "title": entity["name"],
                "source": "entity",
                "wikipedia_url": "",
                "base_score": entity["confidence"],
            }

    candidate_list = list(candidates.values())
    if not candidate_list:
        return []

    # Rank by embedding similarity to article
    texts_to_embed = [article_text] + [c["title"] for c in candidate_list]
    embeddings = embed_texts(texts_to_embed)

    article_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    for i, candidate in enumerate(candidate_list):
        sim = cosine_similarity(article_embedding, candidate_embeddings[i])
        candidate["similarity"] = float(sim * 0.7 + candidate["base_score"] * 0.3)

    candidate_list.sort(key=lambda c: c["similarity"], reverse=True)
    candidate_list = candidate_list[:max_candidates]

    return candidate_list
