"""
Named Entity Recognition + Entity Linking using Azure AI Language.
Replaces Google Cloud NLP from main_2020.py.
"""

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from backend.config import settings


def get_client() -> TextAnalyticsClient:
    return TextAnalyticsClient(
        endpoint=settings.azure_language_endpoint,
        credential=AzureKeyCredential(settings.azure_language_key),
    )


def extract_entities(text: str, min_confidence: float = 0.4) -> list[dict]:
    """
    Extract named entities from text using Azure AI Language NER.
    Returns list of {name, type, confidence} sorted by confidence desc.
    """
    client = get_client()
    result = client.recognize_entities(documents=[{"id": "1", "text": text}])[0]

    if result.is_error:
        raise RuntimeError(f"NER error: {result.error.message}")

    entities = []
    seen = set()
    for entity in result.entities:
        if entity.confidence_score >= min_confidence and entity.text.lower() not in seen:
            seen.add(entity.text.lower())
            entities.append({
                "name": entity.text,
                "type": entity.category,
                "subtype": entity.subcategory or "",
                "confidence": entity.confidence_score,
            })

    # Sort by confidence descending
    entities.sort(key=lambda e: e["confidence"], reverse=True)
    return entities


def extract_entity_links(text: str) -> list[dict]:
    """
    Entity Linking: identifies entities and links them to Wikipedia.
    Returns list of {name, wikipedia_url, confidence, data_source}.
    This directly feeds the referential framework.
    """
    client = get_client()
    result = client.recognize_linked_entities(documents=[{"id": "1", "text": text}])[0]

    if result.is_error:
        raise RuntimeError(f"Entity Linking error: {result.error.message}")

    links = []
    seen = set()
    for entity in result.entities:
        if entity.name.lower() not in seen:
            seen.add(entity.name.lower())
            # Get highest confidence match
            best_match = max(entity.matches, key=lambda m: m.confidence_score)
            links.append({
                "name": entity.name,
                "wikipedia_url": entity.url or "",
                "confidence": best_match.confidence_score,
                "data_source": entity.data_source or "Wikipedia",
            })

    links.sort(key=lambda l: l["confidence"], reverse=True)
    return links
