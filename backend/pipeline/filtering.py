"""
TF-IDF cosine similarity filtering.
Same logic as main_2020.py — filters irrelevant candidate titles.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tfidf_cosine_sim(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity between two texts."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([text1, text2])
        return float((tfidf * tfidf.T).toarray()[0, 1])
    except ValueError:
        return 0.0


def filter_candidates(
    candidates: list[dict],
    article_text: str,
    entities_text: str,
    min_relevance: float = 0.01,
) -> list[dict]:
    """
    Filter candidates using TF-IDF cosine similarity against:
    - The full article text
    - The concatenated entity names

    Removes candidates with zero relevance to both.
    Same purpose as the tf-idf filtering in main_2020.py.
    """
    filtered = []

    for candidate in candidates:
        title = candidate["title"]

        # TF-IDF similarity to article
        sim_article = tfidf_cosine_sim(title, article_text)
        # TF-IDF similarity to entities
        sim_entities = tfidf_cosine_sim(title, entities_text)

        # Combined relevance
        relevance = max(sim_article, sim_entities)

        if relevance >= min_relevance or candidate["source"] == "entity":
            candidate["tfidf_relevance"] = relevance
            candidate["tfidf_article"] = sim_article
            candidate["tfidf_entities"] = sim_entities
            filtered.append(candidate)

    return filtered
