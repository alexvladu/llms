from __future__ import annotations

import re
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - convenience fallback for parser-only use.
    from dataclasses import dataclass

    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]

from city_preprocessing.models import CityArticle, CitySection


def section_id_part(title: str) -> str:
    value = title.lower()
    replacements = str.maketrans(
        {
            "ă": "a",
            "â": "a",
            "î": "i",
            "ș": "s",
            "ş": "s",
            "ț": "t",
            "ţ": "t",
        }
    )
    value = value.translate(replacements)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "sectiune"


def document_id(city_key: str, section_title: str, index: int = 0) -> str:
    return f"{city_key}::{section_id_part(section_title)}::doc_{index:03d}"


def build_profile_text(article: CityArticle) -> str:
    metadata = article.metadata
    lines = [
        f"Oraș: {metadata.get('city', article.display_name)}",
        f"Județ: {metadata['county']}" if metadata.get("county") else None,
        f"Țară: {metadata.get('country', 'România')}",
        f"Tip localitate: {metadata['settlement_type']}" if metadata.get("settlement_type") else None,
        f"Populație 2021: {metadata['population_2021']} locuitori" if metadata.get("population_2021") else None,
        f"Populație 2011: {metadata['population_2011']} locuitori" if metadata.get("population_2011") else None,
        "Transport: are conexiune feroviară" if metadata.get("has_railway") else None,
        f"Drumuri naționale: {', '.join(metadata['national_roads'])}" if metadata.get("national_roads") else None,
        f"Drumuri europene: {', '.join(metadata['european_roads'])}" if metadata.get("european_roads") else None,
        f"Autostrăzi: {', '.join(metadata['motorways'])}" if metadata.get("motorways") else None,
    ]
    return "\n".join(line for line in lines if line)


def base_metadata(article: CityArticle, section: str, source_type: str) -> dict[str, Any]:
    metadata = {
        "city": article.metadata.get("city", article.display_name),
        "city_key": article.city_key,
        "section": section,
        "source_type": source_type,
        "raw_source_path": str(article.source_path),
    }
    for key in ("county", "country", "settlement_type"):
        if key in article.metadata:
            metadata[key] = article.metadata[key]
    return metadata


def build_profile_document(article: CityArticle) -> Document:
    metadata = base_metadata(article, "Profil general", "wikipedia_structured_profile")
    metadata["document_id"] = document_id(article.city_key, "Profil general")
    return Document(page_content=build_profile_text(article), metadata=metadata)


def build_section_document(article: CityArticle, section: CitySection, index: int) -> Document:
    content = f"Oraș: {article.display_name}\nSecțiune: {section.title}\n\n{section.text}"
    metadata = base_metadata(article, section.title, "wikipedia_section")
    metadata["document_id"] = document_id(article.city_key, section.title, index)
    return Document(page_content=content, metadata=metadata)


def build_city_documents(article: CityArticle) -> list[Document]:
    documents = [build_profile_document(article)]
    for index, section in enumerate(article.useful_sections(), start=1):
        documents.append(build_section_document(article, section, index))
    return documents
