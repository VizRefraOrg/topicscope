"""
Named Entity Recognition using spaCy.
No API calls, no size limits, no cost.
"""

import spacy

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


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities using spaCy.
    No document size limit. No API calls.
    """
    nlp = get_nlp()
    doc = nlp(text)

    entity_data = {}
    skip_types = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}

    for ent in doc.ents:
        key = ent.text.strip().lower()
        if len(key) < 2 or ent.label_ in skip_types:
            continue

        if key not in entity_data:
            entity_data[key] = {
                "name": ent.text.strip(),
                "type": ent.label_,
                "count": 0,
                "first_pos": ent.start_char,
            }
        entity_data[key]["count"] += 1

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
        })

    entities.sort(key=lambda e: e["confidence"], reverse=True)
    return entities


def extract_entity_links(text: str) -> list[dict]:
    """Stub — returns empty list. Wikipedia linking disabled for now."""
    return []
