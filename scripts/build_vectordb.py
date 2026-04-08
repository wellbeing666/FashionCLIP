from __future__ import annotations

import argparse

import pandas as pd

from fashionclip.core.embedding import load_embeddings
from fashionclip.core.vector_db import build_chroma_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector database from catalog and embeddings")
    parser.add_argument("--catalog-csv", default="data/catalog.csv")
    parser.add_argument("--embedding-npy", default="data/embeddings.npy")
    parser.add_argument("--persist-dir", default="data/chroma")
    parser.add_argument("--collection", default="fashion_items")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog_csv)
    embeddings = load_embeddings(args.embedding_npy)

    build_chroma_collection(
        catalog=catalog,
        embeddings=embeddings,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    print(f"Vector DB built at: {args.persist_dir}")
    print(f"Collection: {args.collection}")
    print(f"Rows: {len(catalog)}")


if __name__ == "__main__":
    main()
