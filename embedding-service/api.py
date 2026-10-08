"""Minimal CLIP embedding API. Go owns auth, uploads, persistence, and search."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import UnidentifiedImageError

from embedding_engine import EmbeddingEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = EmbeddingEngine()
    yield


app = FastAPI(title="Lynx CLIP Embedding Service", version="0.2.0", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=512)


@app.get("/health")
def health(request: Request) -> dict[str, str | int]:
    engine: EmbeddingEngine = request.app.state.engine
    return {
        "status": "ok",
        "model": "ViT-B-32/laion2b_s34b_b79k",
        "dimension": engine.dimension,
        "device": engine.device,
    }


@app.post("/embed/text")
def embed_text(body: TextRequest, request: Request) -> dict[str, list[float]]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be blank")
    return {"embedding": request.app.state.engine.embed_text(text)}


@app.post("/embed/image")
async def embed_image(request: Request) -> dict[str, list[float]]:
    content = await request.body()
    if not content or len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Provide an image smaller than 10 MB")
    try:
        return {"embedding": request.app.state.engine.embed_image(content)}
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=415, detail="Unsupported image format") from exc
