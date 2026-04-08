from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from fashionclip.core.embedding import ClipEmbedder, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP embeddings for catalog items")
    parser.add_argument("--catalog-csv", default="data/catalog.csv")
    parser.add_argument("--out-npy", default="data/embeddings.npy")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog_csv)
    image_paths = catalog["image_path"].astype(str).tolist()

    embedder = ClipEmbedder(model_name=args.clip_model)
    embeddings = embedder.encode_images(image_paths, batch_size=16)
    save_embeddings(args.out_npy, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {args.out_npy}")


if __name__ == "__main__":
    main()
