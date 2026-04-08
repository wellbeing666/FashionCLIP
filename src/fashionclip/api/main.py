from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from fashionclip.config import settings
from fashionclip.core.agent import StylingAgent
from fashionclip.core.embedding import ClipEmbedder, load_embeddings
from fashionclip.core.evaluator import OutfitEvaluator
from fashionclip.core.retriever import FashionRetriever

app = FastAPI(title="FashionCLIP Stylist API", version="0.1.0")


class AppState:
    embedder: ClipEmbedder | None = None
    retriever: FashionRetriever | None = None
    agent: StylingAgent | None = None


state = AppState()


@app.on_event("startup")
def startup() -> None:
    catalog_path = Path(settings.catalog_csv)
    embedding_path = Path(settings.embedding_npy)

    if not catalog_path.exists() or not embedding_path.exists():
        return

    catalog = pd.read_csv(catalog_path)
    embeddings = load_embeddings(str(embedding_path))

    embedder = ClipEmbedder(settings.clip_model_name)
    retriever = FashionRetriever(catalog=catalog, embeddings=embeddings)
    evaluator = OutfitEvaluator(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
    agent = StylingAgent(retriever=retriever, evaluator=evaluator)

    state.embedder = embedder
    state.retriever = retriever
    state.agent = agent


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "catalog_loaded": state.retriever is not None,
        "llm_enabled": bool(settings.openai_api_key),
    }


@app.post("/recommend")
async def recommend(
    image: UploadFile = File(...),
    base_category: str = Form(...),
    season: str | None = Form(default=None),
    occasion: str | None = Form(default=None),
    gender: str | None = Form(default=None),
    usage: str | None = Form(default=None),
) -> dict[str, Any]:
    if state.embedder is None or state.agent is None:
        raise HTTPException(status_code=400, detail="Catalog or embeddings not loaded. Build index first.")

    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")

    temp_path = Path("data") / "_query_upload.jpg"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(temp_path)

    query_emb = state.embedder.encode_images([str(temp_path)])[0]
    query_item = {
        "item_id": "uploaded",
        "image_path": str(temp_path),
        "category": base_category,
    }

    result = state.agent.run(
        query_embedding=query_emb,
        query_item=query_item,
        season=season,
        occasion=occasion,
        gender=gender,
        usage=usage,
    )
    return result
