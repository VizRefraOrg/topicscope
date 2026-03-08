"""
Named Entity Recognition with computed salience.
Salience approximates Google NLP's salience score using:
  - TF-IDF score (how distinctive is this entity to this document)
  - Frequency (how often mentioned)
  - Position (earlier = more important)
  - Title presence (entities in first sentence get a boost)

The goal is WIDE variation: top entity ~0.3-0.5, bottom entities ~0.001-0.01
This drives mountain size — without wide variation, all mountains look the same.
"""

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_lg")
        except OSError:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _nlp = spacy.load("en_core_web_sm")
    return _nlp


def similar(a: str, b: str) -> bool:
    """Check if two entity names are variations of the same thing."""
    a_low = a.lower().strip().rstrip('.')
    b_low = b.lower().strip().rstrip('.')
    if a_low == b_low:
        return True
    if a_low in b_low or b_low in a_low:
        return True
    a_clean = a_low.replace('.', '').replace(' ', '')
    b_clean = b_low.replace('.', '').replace(' ', '')
    if a_clean == b_clean:
        return True
    if len(a_low) > 3 and len(b_low) > 3 and SequenceMatcher(None, a_low, b_low).ratio() > 0.85:
        return True
    return False


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Merge entities that are variations of the same thing."""
    merged = []
    for entity in entities:
        found = False
        for existing in merged:
            if similar(entity["name"], existing["name"]):
                if len(entity["name"]) > len(existing["name"]):
                    existing["name"] = entity["name"]
                existing["count"] = existing.get("count", 1) + entity.get("count", 1)
                existing["first_pos"] = min(existing.get("first_pos", 9999), entity.get("first_pos", 9999))
                found = True
                break
        if not found:
            merged.append(dict(entity))
    return merged


def compute_tfidf_scores(entity_names: list[str], document_text: str) -> dict:
    """
    Compute TF-IDF score for each entity within the document.
    Higher score = entity is more distinctive/important to this document.
    """
    try:
        # Treat entity names as "queries" against the document
        # Split document into sentences as the corpus
        sentences = [s.strip() for s in document_text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
        if len(sentences) < 2:
            sentences = [document_text]

        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        scores = {}
        for name in entity_names:
            words = name.lower().split()
            total = 0
            count = 0
            for word in words:
                if word in feature_names:
                    idx = list(feature_names).index(word)
                    # Sum TF-IDF across all sentences where this word appears
                    col = tfidf_matrix.getcol(idx).toarray().flatten()
                    total += col.sum()
                    count += 1
            scores[name] = total / max(count, 1)
        return scores
    except Exception:
        return {name: 0.1 for name in entity_names}


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities with computed salience score.
    Salience has WIDE variation (0.001 to 0.5+) like Google NLP.
    """
    nlp = get_nlp()
    doc = nlp(text)
    total_chars = max(len(text), 1)

    # First sentence text (entities here get a boost)
    first_sentence = text[:min(200, len(text))]

    skip_types = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}

    # Collect raw entity data
    entity_data = {}
    for ent in doc.ents:
        key = ent.text.strip().lower().replace('.', '').replace(' ', '')
        if len(ent.text.strip()) < 2 or ent.label_ in skip_types:
            continue

        if key not in entity_data:
            entity_data[key] = {
                "name": ent.text.strip(),
                "type": ent.label_,
                "subtype": "",
                "count": 0,
                "first_pos": ent.start_char,
                "positions": [],
            }
        entity_data[key]["count"] += 1
        entity_data[key]["positions"].append(ent.start_char)
        if len(ent.text.strip()) > len(entity_data[key]["name"]):
            entity_data[key]["name"] = ent.text.strip()

    if not entity_data:
        return []

    entities = list(entity_data.values())

    # Deduplicate
    entities = deduplicate_entities(entities)

    # Compute TF-IDF scores
    entity_names = [e["name"] for e in entities]
    tfidf_scores = compute_tfidf_scores(entity_names, text)

    # ── Compute salience for each entity ─────────────────────────
    # Goal: produce wide variation like Google NLP (0.001 to 0.5)

    max_count = max(e["count"] for e in entities)

    for entity in entities:
        name = entity["name"]

        # 1. TF-IDF score (most important signal)
        tfidf = tfidf_scores.get(name, 0.0)

        # 2. Frequency: log scale so diminishing returns
        freq = np.log1p(entity["count"]) / np.log1p(max_count)  # 0 to 1

        # 3. Position: first mention earlier = more salient
        pos = 1.0 - (entity["first_pos"] / total_chars)  # 1.0 = very start, 0.0 = very end

        # 4. Title/lead boost: entity in first sentence
        title_boost = 1.5 if name.lower() in first_sentence.lower() else 1.0

        # 5. Multi-word boost: "Russian military intelligence" more salient than "Russian"
        word_count_boost = min(len(name.split()) * 0.3, 1.0) + 0.7  # 1.0 to 1.6

        # Combine with weights
        raw_salience = (
            tfidf * 0.35 +
            freq * 0.30 +
            pos * 0.20 +
            0.15  # base
        ) * title_boost * word_count_boost

        entity["salience"] = raw_salience

    # Normalize: make top entity ~0.35, use exponential to widen spread
    max_sal = max(e["salience"] for e in entities)
    if max_sal > 0:
        for e in entities:
            # Normalize to 0-1
            norm = e["salience"] / max_sal
            # Apply power curve to widen spread (top stays ~0.35, bottom drops to ~0.005)
            e["salience"] = float(np.power(norm, 1.8) * 0.35)

    # Also set confidence = salience for backward compatibility
    for e in entities:
        e["confidence"] = e["salience"]
        # Clean up internal fields
        e.pop("positions", None)
        e.pop("first_pos", None)
        e.pop("count", None)

    entities.sort(key=lambda e: e["salience"], reverse=True)
    return entities


def extract_entity_links(text: str) -> list[dict]:
    """Stub — Wikipedia linking disabled."""
    return []
