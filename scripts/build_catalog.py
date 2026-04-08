from __future__ import annotations

import argparse
from pathlib import Path

from fashionclip.data.catalog import build_catalog_from_deepfashion2, build_catalog_from_kaggle_styles


def main() -> None:
    parser = argparse.ArgumentParser(description="Build garment catalog CSV from supported fashion datasets")
    parser.add_argument("--dataset", choices=["deepfashion2", "kaggle_styles"], default="kaggle_styles")
    parser.add_argument("--image-root", required=True, help="Path to image folder")
    parser.add_argument("--anno-root", default=None, help="Path to DeepFashion2 annotation json folder")
    parser.add_argument("--styles-csv", default=None, help="Path to Kaggle styles.csv")
    parser.add_argument("--out-csv", default="data/catalog.csv", help="Output CSV path")
    parser.add_argument("--season", default=None, help="Optional default season label")
    parser.add_argument("--occasion", default=None, help="Optional default occasion label")
    args = parser.parse_args()

    if args.dataset == "deepfashion2":
        if not args.anno_root:
            raise ValueError("--anno-root is required when --dataset deepfashion2")
        df = build_catalog_from_deepfashion2(
            image_root=args.image_root,
            anno_root=args.anno_root,
            out_csv=args.out_csv,
            default_season=args.season,
            default_occasion=args.occasion,
        )
    else:
        if not args.styles_csv:
            raise ValueError("--styles-csv is required when --dataset kaggle_styles")

        styles_path = Path(args.styles_csv)
        if styles_path.is_dir():
            styles_path = styles_path / "styles.csv"
        if not styles_path.exists() or not styles_path.is_file():
            raise ValueError(f"styles csv not found: {styles_path}")

        df = build_catalog_from_kaggle_styles(
            styles_csv=str(styles_path),
            image_root=args.image_root,
            out_csv=args.out_csv,
        )

    print(f"Catalog rows: {len(df)}")
    print(f"Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
