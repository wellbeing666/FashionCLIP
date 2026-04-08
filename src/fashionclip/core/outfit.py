from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fashionclip.core.retriever import FashionRetriever


@dataclass
class OutfitCandidate:
    items: dict[str, dict]
    score: float


CATEGORY_COMPATIBILITY = {
    "top": ["bottom", "outerwear", "shoes"],
    "bottom": ["top", "outerwear", "shoes"],
    "outerwear": ["top", "bottom", "shoes"],
    "shoes": ["top", "bottom", "outerwear"],
    "dress": ["outerwear", "shoes"],
}


def build_outfit_candidates(
    retriever: FashionRetriever,
    query_embedding: np.ndarray,
    query_item: dict,
    season: str | None,
    occasion: str | None,
    gender: str | None = None,
    usage: str | None = None,
    per_category_top_k: int = 5,
    max_outfits: int = 8,
) -> list[OutfitCandidate]:
    base_category = str(query_item.get("category", "other")).lower()
    targets = CATEGORY_COMPATIBILITY.get(base_category, ["top", "bottom", "outerwear", "shoes"])

    bucket: dict[str, pd.DataFrame] = {}
    for cat in targets:
        bucket[cat] = retriever.query_by_embedding(
            query_embedding=query_embedding,
            target_category=cat,
            season=season,
            occasion=occasion,
            gender=gender,
            usage=usage,
            top_k=per_category_top_k,
        )

    # Build simple Cartesian combinations with the best few from each required category.
    candidates: list[OutfitCandidate] = []
    current = {
        "query": {
            "item_id": query_item.get("item_id", "query_item"),
            "image_path": query_item.get("image_path", "uploaded_image"),
            "category": base_category,
            "similarity": 1.0,
        }
    }

    ordered_targets = [t for t in targets if len(bucket[t]) > 0]
    if not ordered_targets:
        return [OutfitCandidate(items=current, score=1.0)]

    def dfs(depth: int, acc: dict[str, dict], score_parts: list[float]) -> None:
        if len(candidates) >= max_outfits:
            return
        if depth == len(ordered_targets):
            mean_score = float(np.mean(score_parts)) if score_parts else 0.0
            candidates.append(OutfitCandidate(items=dict(acc), score=mean_score))
            return

        cat = ordered_targets[depth]
        rows = bucket[cat].head(3).to_dict("records")
        for row in rows:
            acc[cat] = row
            dfs(depth + 1, acc, score_parts + [float(row.get("similarity", 0.0))])
            del acc[cat]

    dfs(0, current, [1.0])
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:max_outfits]
