from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are a professional fashion stylist.
Score each outfit in 0-10 for coherence, occasion match, and season match.
Return strict JSON: {\"ranked\": [{\"index\": int, \"score\": float, \"reason\": str}]}.
"""


class OutfitEvaluator:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.enabled = bool(api_key)
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url) if self.enabled else None

    def evaluate(
        self,
        outfits: list[dict[str, Any]],
        user_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not outfits:
            return []

        if not self.enabled or self.client is None:
            # Fallback: keep retrieval order if no LLM credentials are configured.
            return [
                {
                    "index": i,
                    "score": round(float(o.get("retrieval_score", 0.0)) * 10, 2),
                    "reason": "retrieval-only ranking",
                }
                for i, o in enumerate(outfits)
            ]

        prompt = {
            "context": user_context,
            "outfits": outfits,
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return parsed.get("ranked", [])
