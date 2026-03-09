"""
Named Entity Recognition with wide-range salience.
Uses TF-IDF × frequency² for 100×+ variation between top and bottom entities.
Passes through entity type (tag) for color coding.
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
    merged = []
    for entity in entities:
        found = False
        for existing in merged:
            if similar(entity["name"], existing["name"]):
                if len(entity["name"]) > len(existing["name"]):
                    existing["name"] = entity["name"]
                existing["count"] = existing.get("count", 1) + entity.get("count", 1)
                existing["first_pos"] = min(existing.get("first_pos", 9999), entity.get("first_pos", 9999))
                # Keep the more specific type
                if existing["type"] in ("NORP",) and entity["type"] not in ("NORP",):
                    existing["type"] = entity["type"]
                found = True
                break
        if not found:
            merged.append(dict(entity))
    return merged


def compute_tfidf_scores(entity_names: list[str], document_text: str) -> dict:
    try:
        sentences = [s.strip() for s in document_text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
        if len(sentences) < 2:
            sentences = [document_text]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = list(vectorizer.get_feature_names_out())
        scores = {}
        for name in entity_names:
            words = name.lower().split()
            total = 0
            count = 0
            for word in words:
                if word in feature_names:
                    idx = feature_names.index(word)
                    col = tfidf_matrix.getcol(idx).toarray().flatten()
                    total += col.sum()
                    count += 1
            scores[name] = total / max(count, 1)
        return scores
    except Exception:
        return {name: 0.1 for name in entity_names}


def extract_entities(text: str) -> list[dict]:
    """
    Extract entities with wide-range salience and entity type (tag).
    Salience uses TF-IDF × frequency² for 100×+ variation.
    """
    nlp = get_nlp()
    doc = nlp(text)
    total_chars = max(len(text), 1)
    first_sentence = text[:min(300, len(text))]

    skip_types = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}

    entity_data = {}
    for ent in doc.ents:
        key = ent.text.strip().lower().replace('.', '').replace(' ', '')
        if len(ent.text.strip()) < 2 or ent.label_ in skip_types:
            continue
        if key not in entity_data:
            entity_data[key] = {
                "name": ent.text.strip(),
                "type": ent.label_,  # PERSON, ORG, GPE, etc — this becomes "tag"
                "subtype": "",
                "count": 0,
                "first_pos": ent.start_char,
            }
        entity_data[key]["count"] += 1
        if len(ent.text.strip()) > len(entity_data[key]["name"]):
            entity_data[key]["name"] = ent.text.strip()

    if not entity_data:
        return []

    entities = list(entity_data.values())
    entities = deduplicate_entities(entities)

    # Compute TF-IDF scores
    entity_names = [e["name"] for e in entities]
    tfidf_scores = compute_tfidf_scores(entity_names, text)

    # ── Wide-range salience: TF-IDF × frequency² × position ─────
    # Goal: 100×+ variation (like Google NLP's 0.001 to 0.95)
    max_count = max(e["count"] for e in entities)

    for entity in entities:
        name = entity["name"]
        tfidf = tfidf_scores.get(name, 0.0)

        # frequency² gives much wider spread than linear
        freq_sq = (entity["count"] / max(max_count, 1)) ** 2

        # Position: earlier = more important
        pos = 1.0 - (entity["first_pos"] / total_chars) * 0.5

        # Title/lead boost
        title_boost = 2.0 if name.lower() in first_sentence.lower() else 1.0

        # Multi-word boost
        word_boost = 1.0 + min(len(name.split()) - 1, 3) * 0.3

        # Raw salience: TF-IDF × freq² × position × boosts
        raw = (tfidf * 0.5 + freq_sq * 0.3 + pos * 0.2) * title_boost * word_boost

        entity["salience"] = raw

    # Normalize so top = ~0.95, but DON'T compress the bottom
    # Just scale relative to max, preserving the natural wide spread
    max_sal = max(e["salience"] for e in entities) if entities else 1
    if max_sal > 0:
        for e in entities:
            e["salience"] = float(e["salience"] / max_sal * 0.95)

    # Set confidence = salience for backward compat
    for e in entities:
        e["confidence"] = e["salience"]
        e.pop("first_pos", None)

    entities.sort(key=lambda e: e["salience"], reverse=True)
    return entities


def extract_entity_links(text: str) -> list[dict]:
    return []
