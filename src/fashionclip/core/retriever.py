from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class FashionRetriever:
    def __init__(self, catalog: pd.DataFrame, embeddings: np.ndarray) -> None:
        if len(catalog) != len(embeddings):
            raise ValueError("Catalog size and embedding size must match")
        self.catalog = catalog.reset_index(drop=True)
        self.embeddings = embeddings

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
        candidates = self.catalog.copy()
        idx_mask = np.ones(len(candidates), dtype=bool)

        if target_category:
            idx_mask &= candidates["category"].astype(str).str.lower().eq(target_category.lower()).values
        if season:
            idx_mask &= (
                candidates["season"].fillna("").astype(str).str.lower().eq(season.lower())
                | candidates["season"].isna()
            ).values
        if occasion:
            idx_mask &= (
                candidates["occasion"].fillna("").astype(str).str.lower().eq(occasion.lower())
                | candidates["occasion"].isna()
            ).values
        if gender and "gender" in candidates.columns:
            idx_mask &= (
                candidates["gender"].fillna("").astype(str).str.lower().eq(gender.lower())
                | candidates["gender"].isna()
            ).values
        if usage and "usage" in candidates.columns:
            idx_mask &= (
                candidates["usage"].fillna("").astype(str).str.lower().eq(usage.lower())
                | candidates["usage"].isna()
            ).values

        filtered_idx = np.where(idx_mask)[0]
        if len(filtered_idx) == 0:
            return candidates.iloc[[]]

        filtered_embeddings = self.embeddings[filtered_idx]
        sims = cosine_similarity(query_embedding.reshape(1, -1), filtered_embeddings).reshape(-1)
        ranked = np.argsort(-sims)[:top_k]
        hit_idx = filtered_idx[ranked]

        result = self.catalog.iloc[hit_idx].copy()
        result["similarity"] = sims[ranked]
        return result
