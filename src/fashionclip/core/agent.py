from __future__ import annotations

from typing import Any

from fashionclip.core.evaluator import OutfitEvaluator
from fashionclip.core.outfit import build_outfit_candidates
from fashionclip.core.retriever import FashionRetriever


class StylingAgent:
    def __init__(self, retriever: FashionRetriever, evaluator: OutfitEvaluator) -> None:
        self.retriever = retriever
        self.evaluator = evaluator

    def run(
        self,
        query_embedding,
        query_item: dict[str, Any],
        season: str | None,
        occasion: str | None,
        gender: str | None = None,
        usage: str | None = None,
    ) -> dict[str, Any]:
        candidates = build_outfit_candidates(
            retriever=self.retriever,
            query_embedding=query_embedding,
            query_item=query_item,
            season=season,
            occasion=occasion,
            gender=gender,
            usage=usage,
            per_category_top_k=5,
            max_outfits=8,
        )

        serialized = []
        for c in candidates:
            items = {k: v for k, v in c.items.items()}
            serialized.append({"items": items, "retrieval_score": c.score})

        ranked = self.evaluator.evaluate(
            outfits=serialized,
            user_context={
                "season": season,
                "occasion": occasion,
                "gender": gender,
                "usage": usage,
            },
        )

        # Merge LLM rank with retrieval details.
        by_idx = {int(r["index"]): r for r in ranked if "index" in r}
        merged = []
        for i, outfit in enumerate(serialized):
            llm_meta = by_idx.get(i, {"score": outfit["retrieval_score"] * 10, "reason": "no-llm"})
            merged.append(
                {
                    "outfit": outfit,
                    "llm_score": float(llm_meta.get("score", 0.0)),
                    "reason": llm_meta.get("reason", ""),
                }
            )

        merged.sort(key=lambda x: x["llm_score"], reverse=True)
        return {"top_outfits": merged[:5]}
