"""
TopicScope Backend API
"""

import time
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from backend.config import settings

app = FastAPI(title="TopicScope API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000)


class TopicResult(BaseModel):
    label: str
    x: float
    y: float
    size: float         # sqrt(salience) * 100 — ready to render
    height: float       # 0 to 1 relevance
    tag: str = "MISC"   # entity type: PERSON, ORG, GPE, etc
    cluster: int = 1    # K-means group ID
    salience: float = 0.0
    wikipedia_url: str = ""
    source: str = ""
    similarity: float = 0.0
    shore_markers: list = []
    grid_gx: int = -1
    grid_gy: int = -1


class EntityResult(BaseModel):
    name: str
    type: str
    confidence: float


class AnalyseResponse(BaseModel):
    topics: list[TopicResult]
    entities: list[EntityResult]
    all_entity_positions: list = []
    metadata: dict
    debug: list = []
    distance_matrix: list = []
    heightmap: dict = {}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


async def run_analysis(text: str) -> AnalyseResponse:
    start_time = time.time()
    text = text.strip()
    word_count = len(text.split())

    if word_count < 20:
        raise HTTPException(status_code=400, detail="Text too short. Minimum 20 words required.")
    if word_count > settings.max_word_count:
        raise HTTPException(status_code=400, detail=f"Text too long. Maximum {settings.max_word_count} words.")

    try:
        from backend.pipeline.ner import extract_entities, extract_entity_links
        entities = extract_entities(text)
        entity_links = extract_entity_links(text)

        from backend.pipeline.topic_lookup import find_candidate_topics
        candidates = await find_candidate_topics(
            article_text=text, entities=entities,
            entity_links=entity_links, max_candidates=60,
        )
        if not candidates:
            raise HTTPException(status_code=422, detail="No topics could be discovered.")

        from backend.pipeline.filtering import filter_candidates
        entities_text = " ".join(e["name"] for e in entities)
        candidates = filter_candidates(candidates, text, entities_text)
        if not candidates:
            raise HTTPException(status_code=422, detail="All candidates were filtered out.")

        # Pass entities for tag/salience lookup
        from backend.pipeline.reduction import compute_distance_and_reduce
        result = compute_distance_and_reduce(
            candidates, text,
            entities=entities,
            entities_text=entities_text,
        )
        candidates = result["candidates"]
        debug_data = result.get("debug", [])
        dist_matrix = result.get("distance_matrix", [])
        heightmap_data = result.get("heightmap", {})
        all_entity_positions = result.get("all_entity_positions", [])

        from backend.pipeline.clustering import process_topics
        final_topics = process_topics(candidates)

        elapsed_ms = int((time.time() - start_time) * 1000)

        topics_out = [
            TopicResult(
                label=t["title"], x=t["x"], y=t["y"],
                size=t["size"], height=t["height"],
                tag=t.get("tag", "MISC"),
                cluster=t.get("cluster", 1),
                salience=t.get("salience", 0),
                wikipedia_url=t.get("wikipedia_url", ""),
                source=t.get("source", ""),
                similarity=t.get("similarity", 0.0),
                shore_markers=t.get("shore_markers", []),
                grid_gx=t.get("grid_gx", -1),
                grid_gy=t.get("grid_gy", -1),
            )
            for t in final_topics
        ]

        entities_out = [
            EntityResult(name=e["name"], type=e["type"], confidence=e["confidence"])
            for e in entities[:30]
        ]

        return AnalyseResponse(
            topics=topics_out, entities=entities_out,
            all_entity_positions=all_entity_positions,
            metadata={
                "word_count": word_count, "entities_found": len(entities),
                "candidates_discovered": len(candidates),
                "topics_final": len(final_topics), "processing_time_ms": elapsed_ms,
            },
            debug=debug_data, distance_matrix=dist_matrix,
            heightmap=heightmap_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyse", response_model=AnalyseResponse)
async def analyse(request: AnalyseRequest):
    return await run_analysis(request.text)


@app.post("/api/upload", response_model=AnalyseResponse)
async def upload_file(file: UploadFile = File(...)):
    from backend.pipeline.file_handler import extract_text_from_file
    text = await extract_text_from_file(file)
    return await run_analysis(text)


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _serve_static_file(filename: str, media_type: str):
    path = os.path.join(static_dir, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"{filename} not found")


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    return _serve_static_file("robots.txt", "text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    return _serve_static_file("sitemap.xml", "application/xml")


@app.get("/llms.txt", include_in_schema=False)
async def llms_txt():
    return _serve_static_file("llms.txt", "text/plain")


@app.get("/manifest.json", include_in_schema=False)
async def manifest_json():
    return _serve_static_file("manifest.json", "application/manifest+json")


@app.get("/favicon.png", include_in_schema=False)
async def favicon_png():
    return _serve_static_file("favicon.png", "image/png")


@app.get("/blog", include_in_schema=False)
async def blog_index():
    """Simple index of blog posts — lists all .html files in static/blog/."""
    blog_dir = os.path.join(static_dir, "blog")
    if not os.path.isdir(blog_dir):
        raise HTTPException(status_code=404, detail="no blog posts yet")
    items = []
    for fname in sorted(os.listdir(blog_dir)):
        if fname.endswith(".html"):
            slug = fname[:-5]
            items.append(f'<li><a href="/blog/{slug}">{slug.replace("-", " ").title()}</a></li>')
    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Blog | VizRefra</title>
<meta name="description" content="VizRefra blog — text visualization and topic modeling insights.">
<link rel="canonical" href="https://vizrefra.com/blog">
</head><body><main><h1>Blog</h1><ul>{''.join(items) or '<li>No posts yet.</li>'}</ul></main></body></html>"""
    return Response(content=html, media_type="text/html")


@app.get("/blog/{slug}", include_in_schema=False)
async def blog_post(slug: str):
    """Serve a single blog post by canonical URL (no .html). Strips any .html suffix
    a crawler accidentally appends, and prevents path-traversal."""
    if "/" in slug or ".." in slug:
        raise HTTPException(status_code=400, detail="invalid slug")
    slug = slug.removesuffix(".html")
    path = os.path.join(static_dir, "blog", f"{slug}.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="blog post not found")
    return FileResponse(path, media_type="text/html")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "service": "TopicScope API"}
