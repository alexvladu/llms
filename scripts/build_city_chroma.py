from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import requests
from chromadb.api.types import EmbeddingFunction

from city_preprocessing.chunking import split_city_documents
from city_preprocessing.pipeline import load_city_documents

EMBEDDING_CONFIG_FILENAME = "embedding_config.json"


@dataclass
class Chunk:
    chunk_id: str
    city: str
    city_key: str
    section: str
    source: str
    chunk_index: int
    text: str
    metadata: dict


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalize_whitespace(text)
    if not text:
        return []

    pieces: list[str] = []
    paragraphs = text.split("\n")
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            pieces.append(current)
            carry = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{carry}{paragraph}".strip()
        else:
            start = 0
            step = chunk_size - chunk_overlap
            while start < len(paragraph):
                stop = start + chunk_size
                pieces.append(paragraph[start:stop].strip())
                start += step
            current = ""

        while len(current) > chunk_size:
            pieces.append(current[:chunk_size].strip())
            current = current[chunk_size - chunk_overlap :].strip()

    if current:
        pieces.append(current)

    return [piece for piece in pieces if piece]


def load_chunks(text_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    documents = load_city_documents(text_dir)
    splits = split_city_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for split in splits:
        metadata = dict(split.metadata)
        fallback_digest = hashlib.md5(
            f"{metadata.get('city_key')}:{metadata.get('section')}:{metadata.get('chunk_index')}:{split.page_content}".encode("utf-8")
        ).hexdigest()[:12]
        chunk_id = metadata.get("chunk_id") or f"{metadata.get('city_key', 'unknown')}::{fallback_digest}"
        chunks.append(
            Chunk(
                chunk_id=str(chunk_id),
                city=str(metadata.get("city", "unknown")),
                city_key=str(metadata.get("city_key", "unknown")),
                section=str(metadata.get("section", "unknown")),
                source=str(metadata.get("raw_source_path", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                text=split.page_content,
                metadata=metadata,
            )
        )
    return chunks


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, input: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": input},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, local_files_only: bool = False):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.local_files_only = local_files_only
        self.torch = torch
        self.model_source = snapshot_download(
            repo_id=self.model_name,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=True,
        )
        self.transformer_model = AutoModel.from_pretrained(
            self.model_source,
            local_files_only=True,
        )
        self.transformer_model.eval()
        self.pooling_config = self._load_pooling_config()

    def _load_pooling_config(self) -> dict:
        pooling_path = Path(self.model_source) / "1_Pooling" / "config.json"
        if not pooling_path.exists():
            return {
                "pooling_mode_mean_tokens": True,
                "pooling_mode_cls_token": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
        return json.loads(pooling_path.read_text(encoding="utf-8"))

    def __call__(self, input: list[str]) -> list[list[float]]:
        with self.torch.no_grad():
            encoded = self.tokenizer(
                input,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output = self.transformer_model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = self._pool_embeddings(token_embeddings, attention_mask)
            normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            return normalized.cpu().numpy().tolist()

    def _pool_embeddings(self, token_embeddings, attention_mask):
        if self.pooling_config.get("pooling_mode_cls_token"):
            return token_embeddings[:, 0]

        if self.pooling_config.get("pooling_mode_max_tokens"):
            masked = token_embeddings.masked_fill(attention_mask == 0, float("-inf"))
            return masked.max(dim=1).values

        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)

        if self.pooling_config.get("pooling_mode_mean_sqrt_len_tokens"):
            return summed / counts.sqrt()

        return summed / counts


def embedding_config_path(persist_dir: str | Path, collection: str) -> Path:
    persist_dir = Path(persist_dir)
    safe_collection = collection.replace("/", "_")
    return persist_dir / f"{safe_collection}.{EMBEDDING_CONFIG_FILENAME}"


def build_embedding_config(args: argparse.Namespace, embedding_function: EmbeddingFunction) -> dict:
    config = {
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "local_files_only": bool(getattr(args, "local_files_only", False)),
    }

    if args.embedding_backend == "openai-compatible":
        config["base_url"] = args.base_url
    else:
        config["resolved_model_source"] = getattr(embedding_function, "model_source", args.embedding_model)

    return config


def write_embedding_config(persist_dir: str | Path, collection: str, config: dict) -> Path:
    path = embedding_config_path(persist_dir, collection)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_embedding_config(persist_dir: str | Path, collection: str) -> dict | None:
    path = embedding_config_path(persist_dir, collection)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_embedding_function(args: argparse.Namespace) -> EmbeddingFunction:
    if args.embedding_backend == "openai-compatible":
        return OpenAICompatibleEmbeddingFunction(
            base_url=args.base_url,
            model=args.embedding_model,
            api_key=args.api_key,
        )

    return SentenceTransformerEmbeddingFunction(
        args.embedding_model,
        local_files_only=args.local_files_only,
    )


def apply_embedding_config_to_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    args.embedding_backend = config["embedding_backend"]
    args.embedding_model = config["embedding_model"]
    args.local_files_only = config.get("local_files_only", False)
    if args.embedding_backend == "openai-compatible":
        args.base_url = config.get("base_url", args.base_url)
    return args


def batched(items: list[Chunk], batch_size: int) -> Iterable[list[Chunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess, section, chunk, and load Romanian city documents into Chroma.")
    parser.add_argument("--text-dir", default="cities_text", help="Folder with one .txt file per city.")
    parser.add_argument("--persist-dir", default="chroma_cities", help="Folder where Chroma will persist data.")
    parser.add_argument("--collection", default="romanian_cities", help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=1400, help="Target size in characters for each chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Character overlap between consecutive chunks.")
    parser.add_argument("--batch-size", type=int, default=32, help="How many chunks to embed per request.")
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "openai-compatible"],
        default="sentence-transformers",
        help="Embedding source.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already cached sentence-transformers files.",
    )
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="Base URL for an OpenAI-compatible embedding server.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key for an OpenAI-compatible embedding server.")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection before loading.")
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    if not text_dir.exists():
        print(f"Text directory not found: {text_dir}", file=sys.stderr)
        return 1

    print(f"Loading city files from {text_dir.resolve()}")
    chunks = load_chunks(text_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        print("No chunks were created. Check the input files.", file=sys.stderr)
        return 1

    total_chars = sum(len(chunk.text) for chunk in chunks)
    avg_chunk_size = math.floor(total_chars / len(chunks))
    print(f"Prepared {len(chunks)} section-aware chunks from {len(list(text_dir.glob('*.txt')))} city files")
    print(f"Average chunk size: {avg_chunk_size} chars")

    print(f"Using embeddings backend: {args.embedding_backend}")
    print(f"Embedding model: {args.embedding_model}")
    embedding_function = make_embedding_function(args)
    config = build_embedding_config(args, embedding_function)

    client = chromadb.PersistentClient(path=args.persist_dir)

    if args.reset:
        existing = {collection.name for collection in client.list_collections()}
        if args.collection in existing:
            print(f"Deleting existing collection: {args.collection}")
            client.delete_collection(args.collection)

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[], limit=100000)["ids"])
    pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]

    if not pending_chunks:
        config_path = write_embedding_config(args.persist_dir, args.collection, config)
        print("Collection already contains all chunks. Nothing new to add.")
        print(f"Collection count: {collection.count()}")
        print(f"Embedding config: {config_path}")
        return 0

    print(f"Adding {len(pending_chunks)} new chunks to Chroma at {Path(args.persist_dir).resolve()}")
    for batch_index, batch in enumerate(batched(pending_chunks, args.batch_size), start=1):
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[
                chunk.metadata
                for chunk in batch
            ],
        )
        print(f"  Batch {batch_index}: stored {len(batch)} chunks")

    config_path = write_embedding_config(args.persist_dir, args.collection, config)
    print(f"Done. Collection '{args.collection}' now contains {collection.count()} chunks.")
    print(f"Embedding config saved to {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
