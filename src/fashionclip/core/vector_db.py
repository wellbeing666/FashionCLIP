from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd


def _clean_filter_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"null", "none", "nan", "undefined"}:
        return None
    return normalized


def _build_where_clause(filters: dict[str, str | None]) -> dict[str, Any] | None:
    clauses: list[dict[str, str]] = []
    for key, raw in filters.items():
        cleaned = _clean_filter_value(raw)
        if cleaned is None:
            continue
        clauses.append({key: cleaned})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaFashionRetriever:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        catalog: pd.DataFrame,
    ) -> None:
        self.catalog = catalog.copy().reset_index(drop=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self._id_to_row = {}
        for _, row in self.catalog.iterrows():
            item_id = str(row.get("item_id", ""))
            if item_id:
                self._id_to_row[item_id] = row.to_dict()

    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        target_category: str | None = None,
        season: str | None = None,
        occasion: str | None = None,
        gender: str | None = None,
        usage: str | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        where = _build_where_clause(
            {
                "category": target_category,
                "season": season,
                "occasion": occasion,
                "gender": gender,
                "usage": usage,
            }
        )

        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        if isinstance(query_vec[0], float):
            query_vec = [query_vec]

        result = self.collection.query(
            query_embeddings=query_vec,
            n_results=max(top_k, 1),
            where=where,
            include=["distances", "metadatas"],
        )

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not ids:
            return pd.DataFrame([])

        rows: list[dict[str, Any]] = []
        for i, item_id in enumerate(ids):
            row = dict(self._id_to_row.get(str(item_id), {}))
            if not row:
                continue
            distance = float(distances[i]) if i < len(distances) else 1.0
            row["similarity"] = 1.0 - distance
            rows.append(row)

        return pd.DataFrame(rows)


def build_chroma_collection(
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    persist_dir: str,
    collection_name: str,
) -> None:
    if len(catalog) != len(embeddings):
        raise ValueError("Catalog size and embedding size must match for vector DB ingestion")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for i, row in catalog.reset_index(drop=True).iterrows():
        item_id = str(row.get("item_id", i))
        ids.append(item_id)

        metadata: dict[str, Any] = {}
        for k, v in row.to_dict().items():
            if pd.isna(v):
                continue
            # Chroma metadata expects scalar json-like values.
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v
            else:
                metadata[k] = str(v)
        metadatas.append(metadata)
        vectors.append(embeddings[i].astype(float).tolist())

    batch_size = 512
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
        )
