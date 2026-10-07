"""FastAPI endpoints for Lynx semantic image retrieval."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retrieval_engine import RetrievalEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the CLIP model and vector index once when the service starts,
    # instead of loading them for every HTTP request.
    app.state.engine = RetrievalEngine()
    yield


app = FastAPI(title="Lynx Embedding Service", version="0.1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    filename: str
    image_url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/health")
def health(request: Request) -> dict[str, int | str]:
    engine: RetrievalEngine = request.app.state.engine
    return {
        "status": "ok",
        "model": engine.model_name,
        "embedding_dimension": engine.embedding_dimension,
        "device": engine.device,
        "indexed_images": engine.image_count,
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    engine: RetrievalEngine = request.app.state.engine
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must contain non-whitespace text")
    results = engine.search(query, body.top_k)
    return SearchResponse(query=query, results=results)


@app.get("/images/{filename}")
def get_image(filename: str, request: Request) -> FileResponse:
    engine: RetrievalEngine = request.app.state.engine
    image_path = engine.image_paths.get(filename)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")
