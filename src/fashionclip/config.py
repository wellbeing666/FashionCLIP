from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    catalog_csv: str = os.getenv("CATALOG_CSV", "data/catalog.csv")
    embedding_npy: str = os.getenv("EMBEDDING_NPY", "data/embeddings.npy")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "data/chroma")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "fashion_items")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


settings = Settings()
