from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import chromadb

from build_city_chroma import (
    OpenAICompatibleEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
    apply_embedding_config_to_args,
    read_embedding_config,
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_collection_cities(collection) -> dict[str, str]:
    data = collection.get(include=["metadatas"], limit=100000)
    cities = {
        meta["city_key"]: meta.get("city", meta["city_key"])
        for meta in data["metadatas"]
        if meta and "city_key" in meta
    }
    return cities


def detect_city_filter(query: str, cities: dict[str, str]) -> str | None:
    normalized_query = f" {normalize_text(query)} "
    aliases: dict[str, set[str]] = {}

    for city_key, city in cities.items():
        normalized_city = normalize_text(city)
        normalized_key = normalize_text(city_key)
        aliases.setdefault(normalized_city, set()).add(city_key)
        aliases.setdefault(normalized_key, set()).add(city_key)

        parts = normalized_city.split()
        if len(parts) > 1 and len(parts[0]) >= 4:
            aliases.setdefault(parts[0], set()).add(city_key)

    matches: set[str] = set()
    for alias, mapped_cities in aliases.items():
        if f" {alias} " in normalized_query:
            matches.update(mapped_cities)

    if len(matches) == 1:
        return next(iter(matches))
    return None


def detect_section_filter(query: str) -> str | None:
    normalized_query = normalize_text(query)
    section_keywords = [
        ("Demografie", r"\b(populatie|populatia|locuitori|recensamant|demografie)\b"),
        ("Transport", r"\b(tren|gara|drum|dn|transport|feroviar|sosea)\b"),
        ("Educație", r"\b(scoala|scoli|liceu|licee|gradinita|educatie)\b"),
        ("Sănătate", r"\b(spital|sanatate|ambulanta|policlinica)\b"),
        ("Istorie", r"\b(istorie|fondat|atestat|documentar)\b"),
        ("Geografie", r"\b(relief|clima|rau|geografie|temperatura)\b"),
    ]
    for section, pattern in section_keywords:
        if re.search(pattern, normalized_query):
            return section
    return None


def combine_where(filters: list[dict]) -> dict | None:
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def query_once(collection, query: str, top_k: int, where: dict | None) -> list[tuple[str, dict, float]]:
    result = collection.query(query_texts=[query], n_results=top_k, where=where)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return list(zip(documents, metadatas, distances))


def prioritized_query(
    collection,
    query: str,
    top_k: int,
    detected_city: str | None,
    detected_section: str | None,
) -> tuple[list[tuple[str, dict, float]], list[str]]:
    base_filters = [{"city_key": detected_city}] if detected_city else []
    planned_queries: list[tuple[str, dict | None]] = []

    if detected_section:
        planned_queries.append(
            (
                f"section={detected_section}",
                combine_where([*base_filters, {"section": detected_section}]),
            )
        )

    if detected_city:
        planned_queries.append(
            (
                "structured profile",
                combine_where([*base_filters, {"section": "Profil general"}]),
            )
        )
        planned_queries.append(("city-wide fill", combine_where(base_filters)))

    if not planned_queries:
        planned_queries.append(("semantic search", None))

    seen_ids: set[str] = set()
    combined: list[tuple[str, dict, float]] = []
    plan_labels: list[str] = []

    for label, where in planned_queries:
        plan_labels.append(label)
        for document, metadata, distance in query_once(collection, query, top_k, where):
            stable_id = metadata.get("chunk_id") or metadata.get("document_id") or f"{metadata}:{document[:80]}"
            if stable_id in seen_ids:
                continue
            seen_ids.add(stable_id)
            combined.append((document, metadata, distance))
            if len(combined) >= top_k:
                return combined, plan_labels

    return combined[:top_k], plan_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a semantic search query against the Romanian cities Chroma collection.")
    parser.add_argument("query", help="Search text")
    parser.add_argument("--persist-dir", default="chroma_cities", help="Folder where Chroma persists data.")
    parser.add_argument("--collection", default="romanian_cities", help="Chroma collection name.")
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
        default=True,
        help="Use only already cached sentence-transformers files.",
    )
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="Base URL for an OpenAI-compatible embedding server.")
    parser.add_argument("--api-key", default=None, help="API key for an OpenAI-compatible embedding server.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of matches to return.")
    parser.add_argument(
        "--ignore-saved-config",
        action="store_true",
        help="Do not reuse the embedding config saved during indexing.",
    )
    parser.add_argument(
        "--no-auto-city-filter",
        action="store_true",
        help="Do not constrain results when the query explicitly names a known city.",
    )
    parser.add_argument(
        "--no-auto-section-filter",
        action="store_true",
        help="Do not constrain results when the query clearly asks about a known section.",
    )
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise SystemExit(f"Persist directory not found: {persist_dir}")

    saved_config = None
    if not args.ignore_saved_config:
        saved_config = read_embedding_config(args.persist_dir, args.collection)
        if saved_config:
            args = apply_embedding_config_to_args(args, saved_config)

    if args.embedding_backend == "openai-compatible":
        embedding_function = OpenAICompatibleEmbeddingFunction(
            base_url=args.base_url,
            model=args.embedding_model,
            api_key=args.api_key,
        )
    else:
        embedding_function = SentenceTransformerEmbeddingFunction(
            args.embedding_model,
            local_files_only=args.local_files_only,
        )

    if saved_config:
        print(
            "Using saved embedding config: "
            f"backend={args.embedding_backend}, model={args.embedding_model}"
        )

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=args.collection, embedding_function=embedding_function)

    detected_city = None
    detected_section = None
    if not args.no_auto_city_filter:
        cities = load_collection_cities(collection)
        detected_city = detect_city_filter(args.query, cities)
    if not args.no_auto_section_filter:
        detected_section = detect_section_filter(args.query)

    matches, plan_labels = prioritized_query(
        collection,
        args.query,
        args.top_k,
        detected_city,
        detected_section,
    )

    print(f"Query: {args.query}")
    if detected_city:
        print(f"Detected city_key filter: {detected_city}")
    if detected_section:
        print(f"Detected section filter: {detected_section}")
    print(f"Retrieval plan: {' -> '.join(plan_labels)}")
    for idx, (doc, metadata, distance) in enumerate(matches, start=1):
        city = metadata.get("city", "unknown")
        section = metadata.get("section", "unknown")
        preview = doc[:180].replace("\n", " ")
        print(f"[{idx}] city={city} section={section} distance={distance:.4f}")
        print(f"    {preview}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
