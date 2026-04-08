from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEEPFASHION2_CATEGORY_MAP = {
    1: "short_sleeved_shirt",
    2: "long_sleeved_shirt",
    3: "short_sleeved_outwear",
    4: "long_sleeved_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeved_dress",
    11: "long_sleeved_dress",
    12: "vest_dress",
    13: "sling_dress",
}


def normalize_category(raw_category: str) -> str:
    c = raw_category.lower()
    if "shirt" in c or "vest" in c or "sling" in c:
        return "top"
    if "trousers" in c or "shorts" in c or "skirt" in c:
        return "bottom"
    if "outwear" in c or "jacket" in c or "coat" in c:
        return "outerwear"
    if "shoe" in c or "sneaker" in c or "boot" in c or "heel" in c:
        return "shoes"
    if "dress" in c:
        return "dress"
    return "other"


def resolve_kaggle_image_path(image_root: Path, item_id: str) -> Path | None:
    candidates = [
        image_root / f"{item_id}.jpg",
        image_root / f"{item_id}.jpeg",
        image_root / f"{item_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_catalog_from_deepfashion2(
    image_root: str,
    anno_root: str,
    out_csv: str,
    default_season: str | None = None,
    default_occasion: str | None = None,
) -> pd.DataFrame:
    image_root_p = Path(image_root)
    anno_root_p = Path(anno_root)

    rows: list[dict] = []
    for anno_file in anno_root_p.glob("*.json"):
        with anno_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        image_name = payload.get("file_name") or f"{anno_file.stem}.jpg"
        image_path = image_root_p / image_name
        if not image_path.exists():
            fallback_png = image_root_p / f"{anno_file.stem}.png"
            image_path = fallback_png if fallback_png.exists() else image_path

        item_entries = payload.get("item", {})
        if not isinstance(item_entries, dict):
            continue

        for item_key, item_value in item_entries.items():
            cat_id = item_value.get("category_id")
            if cat_id is None:
                continue
            category_name = DEEPFASHION2_CATEGORY_MAP.get(int(cat_id), "unknown")
            rows.append(
                {
                    "item_id": f"{anno_file.stem}_{item_key}",
                    "image_path": str(image_path),
                    "raw_category": category_name,
                    "category": normalize_category(category_name),
                    "season": default_season,
                    "occasion": default_occasion,
                    "color": None,
                    "style": None,
                    "source": "deepfashion2",
                }
            )

    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def build_catalog_from_kaggle_styles(
    styles_csv: str,
    image_root: str,
    out_csv: str,
) -> pd.DataFrame:
    styles = pd.read_csv(styles_csv, on_bad_lines="skip")
    image_root_p = Path(image_root)

    rows: list[dict] = []
    for _, r in styles.iterrows():
        item_id = str(r.get("id", "")).strip()
        if not item_id:
            continue

        image_path = resolve_kaggle_image_path(image_root_p, item_id)
        if image_path is None:
            continue

        article_type = str(r.get("articleType", "unknown"))
        sub_category = str(r.get("subCategory", "unknown"))
        raw_category = article_type if article_type and article_type != "nan" else sub_category
        usage = r.get("usage")

        rows.append(
            {
                "item_id": item_id,
                "image_path": str(image_path),
                "raw_category": raw_category,
                "category": normalize_category(raw_category),
                "gender": r.get("gender"),
                "master_category": r.get("masterCategory"),
                "sub_category": sub_category,
                "article_type": article_type,
                "season": r.get("season"),
                "year": r.get("year"),
                "usage": usage,
                # Keep compatibility with existing API field name.
                "occasion": usage,
                "color": r.get("baseColour"),
                "style": r.get("productDisplayName"),
                "source": "kaggle-fashion-product-images-small",
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_catalog(catalog_csv: str) -> pd.DataFrame:
    return pd.read_csv(catalog_csv)
