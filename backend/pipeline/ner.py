"""
Named Entity Recognition using spaCy with deduplication.
Merges variations like Russian/Russia, UK/U.K./Britain into single entities.
"""

import spacy
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

    # Exact match after lowering
    if a_low == b_low:
        return True

    # One contains the other (e.g. "Russian" contains "Russia")
    if a_low in b_low or b_low in a_low:
        return True

    # Remove periods and compare (U.K. vs UK, G.R.U. vs GRU)
    a_clean = a_low.replace('.', '').replace(' ', '')
    b_clean = b_low.replace('.', '').replace(' ', '')
    if a_clean == b_clean:
        return True

    # High string similarity
    if SequenceMatcher(None, a_low, b_low).ratio() > 0.85:
        return True

    return False


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Merge entities that are variations of the same thing.
    Keeps the longest/most descriptive name.
    """
    merged = []

    for entity in entities:
        found_match = False
        for existing in merged:
            if similar(entity["name"], existing["name"]):
                # Keep the longer/more descriptive name
                if len(entity["name"]) > len(existing["name"]):
                    existing["name"] = entity["name"]
                # Keep highest confidence
                existing["confidence"] = max(existing["confidence"], entity["confidence"])
                existing["count"] = existing.get("count", 1) + entity.get("count", 1)
                found_match = True
                break

        if not found_match:
            merged.append(dict(entity))

    return merged


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities using spaCy with deduplication.
    """
    nlp = get_nlp()
    doc = nlp(text)

    entity_data = {}
    skip_types = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}

    for ent in doc.ents:
        key = ent.text.strip().lower().replace('.', '').replace(' ', '')
        if len(ent.text.strip()) < 2 or ent.label_ in skip_types:
            continue

        if key not in entity_data:
            entity_data[key] = {
                "name": ent.text.strip(),
                "type": ent.label_,
                "count": 0,
                "first_pos": ent.start_char,
            }
        entity_data[key]["count"] += 1
        # Keep the longest surface form
        if len(ent.text.strip()) > len(entity_data[key]["name"]):
            entity_data[key]["name"] = ent.text.strip()

    total_chars = max(len(text), 1)
    entities = []
    for data in entity_data.values():
        pos_score = 1.0 - (data["first_pos"] / total_chars) * 0.3
        freq_score = min(data["count"] / 5.0, 1.0)
        confidence = round(freq_score * 0.6 + pos_score * 0.4, 3)

        entities.append({
            "name": data["name"],
            "type": data["type"],
            "subtype": "",
            "confidence": confidence,
            "count": data["count"],
        })

    # Deduplicate similar entities
    entities = deduplicate_entities(entities)

    entities.sort(key=lambda e: e["confidence"], reverse=True)
    return entities


def extract_entity_links(text: str) -> list[dict]:
    """Stub — returns empty list. Wikipedia linking disabled for now."""
    return []
