"""
Topic lookup: discovers candidate topics from Wikipedia.
Replaces Doc2Vec.most_similar() from main_2020.py.

Phase 1: Uses Wikipedia API search + embedding comparison.
Phase 3: Will upgrade to Azure AI Search vector index.
"""

import httpx
import asyncio
from backend.pipeline.embeddings import embed_texts, cosine_similarity
import numpy as np


async def search_wikipedia(query: str, limit: int = 10) -> list[dict]:
    """
    Search Wikipedia for articles related to the query.
    Returns list of {title, snippet, url}.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "srprop": "snippet",
    }

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


async def find_candidate_topics(
    article_text: str,
    entities: list[dict],
    entity_links: list[dict],
    max_candidates: int = 60,
) -> list[dict]:
    """
    Find candidate topics from multiple sources:
    1. Entity-linked Wikipedia articles (from Azure AI Language)
    2. Wikipedia search using article keywords
    3. Wikipedia search using top entities

    Returns list of {title, source, wikipedia_url, similarity}.
    """
    candidates = {}

    # Source 1: Entity-linked Wikipedia articles (highest quality)
    for link in entity_links:
        title = link["name"]
        if title.lower() not in candidates:
            candidates[title.lower()] = {
                "title": title,
                "source": "entity_link",
                "wikipedia_url": link.get("wikipedia_url", ""),
                "base_score": link["confidence"],
            }

    # Source 2: Wikipedia search using top entities as queries
    top_entities = [e["name"] for e in entities[:8]]

    search_tasks = []
    for entity_name in top_entities:
        search_tasks.append(search_wikipedia(entity_name, limit=8))

    # Also search with combinations of top 2 entities
    if len(top_entities) >= 2:
        combo_query = f"{top_entities[0]} {top_entities[1]}"
        search_tasks.append(search_wikipedia(combo_query, limit=10))

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
                    "base_score": 0.5,  # neutral score, will be refined by embedding
                }

    # Source 3: Entities themselves as candidate topics
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

    # Now rank candidates by embedding similarity to the article
    if len(candidate_list) == 0:
        return []

    # Embed article and all candidate titles
    texts_to_embed = [article_text] + [c["title"] for c in candidate_list]
    embeddings = embed_texts(texts_to_embed)

    article_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    # Score each candidate
    for i, candidate in enumerate(candidate_list):
        sim = cosine_similarity(article_embedding, candidate_embeddings[i])
        # Blend embedding similarity with base score
        candidate["similarity"] = float(sim * 0.7 + candidate["base_score"] * 0.3)
        candidate["embedding_index"] = i

    # Sort by similarity and take top N
    candidate_list.sort(key=lambda c: c["similarity"], reverse=True)
    candidate_list = candidate_list[:max_candidates]

    return candidate_list
