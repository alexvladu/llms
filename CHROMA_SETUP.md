# Romanian Cities Chroma Setup

You do not need a separate Chroma server to build the vector database in this repo. Chroma can persist locally into `chroma_cities/`.

What you do need is an embedding source. The scripts in this repo support two options:

1. `sentence-transformers`
   This runs locally in Python and is the easiest way to build the database end to end.

2. `openai-compatible`
   This matches the notebook design and works with tools like LM Studio if they expose `/v1/embeddings`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build the vector DB locally

```bash
source .venv/bin/activate
python scripts/build_city_chroma.py --reset
```

This will:

- read every file in `cities_text/*.txt`
- normalize and section each Wikipedia-derived city page
- build one structured profile document plus one document per useful section
- chunk the section-aware documents into overlapping text chunks
- embed them with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- persist them into `chroma_cities/` with metadata such as `city_key`, `county`, `section`, and `source_type`

## Write preprocessing artifacts

To generate inspectable JSON, cleaned text, and a global quality report:

```bash
source .venv/bin/activate
python -m city_preprocessing.pipeline \
  --input-dir cities_text \
  --json-output-dir informations/cities_json \
  --clean-text-output-dir informations/cities_clean_text \
  --report-path informations/preprocessing_report.json \
  --overwrite
```

## Build using a local embedding server

If you want to stay aligned with `main.ipynb`, start your local model server first and then run:

```bash
source .venv/bin/activate
python scripts/build_city_chroma.py \
  --reset \
  --embedding-backend openai-compatible \
  --embedding-model text-embedding-nomic-embed-text-v1.5 \
  --base-url http://localhost:1234/v1
```

## Query the stored chunks

```bash
source .venv/bin/activate
python scripts/query_city_chroma.py "atractii turistice in cluj" --top-k 5 --local-files-only
```

## Optional: run Chroma as a server

Only do this if another app needs to connect over HTTP.

```bash
source .venv/bin/activate
chroma run --path ./chroma_cities
```

Default URL:

- `http://localhost:8000`

## Important notebook fixes

The notebook now tries:

- `informations/cities_text/`
- then falls back to `cities_text/`

It also uses `city_preprocessing.pipeline.load_city_documents()` so chunks are created after cleaning and section detection, not from raw city files.
