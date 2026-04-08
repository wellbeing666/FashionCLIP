# FashionCLIP Outfit Recommender

This project builds a garment database from datasets like Kaggle Fashion Product Images (Small) and DeepFashion2, then performs image-based retrieval and outfit composition with optional LLM reranking.

## 1) What it does

- Build a garment catalog from Kaggle `styles.csv` + product images
- Create CLIP image embeddings for all catalog items
- Upload one garment image and provide natural language context (season, occasion)
- Retrieve top matching complementary categories (bottom, shoes, outerwear, etc.)
- Generate outfit candidates and return top-5
- Optionally use an LLM to evaluate and rerank combinations

## 2) Project structure

- `scripts/build_catalog.py`: build catalog csv from Kaggle styles or DeepFashion2
- `scripts/build_embeddings.py`: compute CLIP embeddings
- `src/fashionclip/api/main.py`: FastAPI service
- `src/fashionclip/core/outfit.py`: outfit candidate generation
- `src/fashionclip/core/evaluator.py`: LLM-based reranking
- `src/fashionclip/core/agent.py`: simple agent orchestration

## 3) Install

```bash
pip install -e .
```

## 4) Configure environment

Copy `.env.example` to `.env` and fill keys if you want LLM ranking.

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=gpt-4.1-mini
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CATALOG_CSV=data/catalog.csv
EMBEDDING_NPY=data/embeddings.npy
```

## 5) Build catalog from Kaggle Fashion Product Images (Small)

Expected structure:

- `styles.csv`
- `images/` (file names like `10000.jpg`)

```bash
python scripts/build_catalog.py --dataset kaggle_styles --styles-csv <styles_csv_path> --image-root <images_dir> --out-csv data/catalog.csv
```

Optional: still support DeepFashion2

```bash
python scripts/build_catalog.py --dataset deepfashion2 --image-root <images_dir> --anno-root <annos_dir> --out-csv data/catalog.csv --season spring --occasion commute
```

## 6) Build embeddings

```bash
python scripts/build_embeddings.py --catalog-csv data/catalog.csv --out-npy data/embeddings.npy
```

## 7) Start API

```bash
uvicorn fashionclip.api.main:app --reload
```

## 8) Test recommendation API

Use multipart form data:

- `image`: uploaded garment image file
- `base_category`: one of `top`, `bottom`, `outerwear`, `shoes`, `dress`
- `season`: optional (e.g. spring, summer)
- `occasion`: optional (mapped to dataset usage)
- `gender`: optional (e.g. Men, Women)
- `usage`: optional (e.g. Casual, Sports)

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -F "image=@./demo/top.jpg" \
  -F "base_category=top" \
  -F "season=spring" \
  -F "occasion=casual" \
  -F "gender=Men" \
  -F "usage=Casual"
```

## Notes

- Kaggle styles metadata includes `gender/masterCategory/subCategory/articleType/baseColour/season/year/usage/productDisplayName`, which are now stored in catalog for filtering and reranking context.
- Current composition strategy is retrieval-first and then LLM reranking. You can replace it with graph/constraint search later.
