"""
TopicScope Backend API
FastAPI application that orchestrates the NLP pipeline.
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.config import settings

app = FastAPI(
    title="TopicScope API",
    description="Textual Data Visual Analysis — Topic Discovery & Terrain Visualization",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],  # permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────

class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000, description="Article text to analyse")


class TopicResult(BaseModel):
    label: str
    x: float
    y: float
    size: float
    height: float
    wikipedia_url: str = ""
    source: str = ""
    similarity: float = 0.0


class EntityResult(BaseModel):
    name: str
    type: str
    confidence: float


class AnalyseResponse(BaseModel):
    topics: list[TopicResult]
    entities: list[EntityResult]
    metadata: dict


# ─── Health Check ────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "TopicScope API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ─── Main Analysis Endpoint ──────────────────────────────────────────

@app.post("/api/analyse", response_model=AnalyseResponse)
async def analyse(request: AnalyseRequest):
    """
    Main analysis endpoint.
    Accepts article text, runs the full NLP pipeline, returns topics for terrain visualization.
    """
    start_time = time.time()
    text = request.text.strip()
    word_count = len(text.split())

    if word_count < 20:
        raise HTTPException(status_code=400, detail="Text too short. Minimum 20 words required.")

    if word_count > settings.max_word_count:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.max_word_count} words allowed.",
        )

    try:
        # ─── Step 1: NER + Entity Linking ────────────────────
        from backend.pipeline.ner import extract_entities, extract_entity_links

        entities = extract_entities(text)
        entity_links = extract_entity_links(text)

        # ─── Step 2: Find candidate topics from Wikipedia ────
        from backend.pipeline.topic_lookup import find_candidate_topics

        candidates = await find_candidate_topics(
            article_text=text,
            entities=entities,
            entity_links=entity_links,
            max_candidates=60,
        )

        if not candidates:
            raise HTTPException(status_code=422, detail="No topics could be discovered from this text.")

        # ─── Step 3: TF-IDF filtering ────────────────────────
        from backend.pipeline.filtering import filter_candidates

        entities_text = " ".join(e["name"] for e in entities)
        candidates = filter_candidates(candidates, text, entities_text)

        if not candidates:
            raise HTTPException(status_code=422, detail="All candidates were filtered out. Try different text.")

        # ─── Step 4: Distance matrix + PCA + spacing ─────────
        from backend.pipeline.reduction import compute_distance_and_reduce

        candidates = compute_distance_and_reduce(candidates, text)

        # ─── Step 5: Elimination + Integration ───────────────
        from backend.pipeline.clustering import process_topics

        final_topics = process_topics(candidates)

        # ─── Build response ──────────────────────────────────
        elapsed_ms = int((time.time() - start_time) * 1000)

        topics_out = [
            TopicResult(
                label=t["title"],
                x=t["x"],
                y=t["y"],
                size=t["size"],
                height=t["height"],
                wikipedia_url=t.get("wikipedia_url", ""),
                source=t.get("source", ""),
                similarity=t.get("similarity", 0.0),
            )
            for t in final_topics
        ]

        entities_out = [
            EntityResult(
                name=e["name"],
                type=e["type"],
                confidence=e["confidence"],
            )
            for e in entities[:20]
        ]

        return AnalyseResponse(
            topics=topics_out,
            entities=entities_out,
            metadata={
                "word_count": word_count,
                "entities_found": len(entities),
                "candidates_discovered": len(candidates),
                "topics_final": len(final_topics),
                "processing_time_ms": elapsed_ms,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
