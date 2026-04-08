from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GarmentItem:
    item_id: str
    image_path: str
    category: str
    season: str | None = None
    occasion: str | None = None
    color: str | None = None
    style: str | None = None
    source: str | None = None
