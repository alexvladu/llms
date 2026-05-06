from __future__ import annotations

from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - lightweight fallback for environments without LangChain splitters.
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from city_preprocessing.document_builder import Document

from city_preprocessing.document_builder import section_id_part

DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    separator = next((sep for sep in separators if sep and sep in text), "")
    if separator:
        pieces = text.split(separator)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}{separator}{piece}".strip() if current else piece.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece.strip()
        if current:
            chunks.append(current)
    else:
        step = chunk_size - chunk_overlap
        chunks = [text[start : start + chunk_size].strip() for start in range(0, len(text), step)]
    return [chunk for chunk in chunks if chunk]


def split_document_text(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        return splitter.split_text(text)
    return _fallback_split_text(text, chunk_size, chunk_overlap, separators)


def build_chunk_id(metadata: dict[str, Any], chunk_index: int) -> str:
    city_key = metadata.get("city_key", "unknown")
    section = section_id_part(str(metadata.get("section", "section")))
    return f"{city_key}::{section}::chunk_{chunk_index:03d}"


def split_city_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[Document]:
    separators = separators or DEFAULT_SEPARATORS
    splits: list[Document] = []
    for document in documents:
        chunk_texts = split_document_text(document.page_content, chunk_size, chunk_overlap, separators)
        chunk_count = len(chunk_texts)
        for index, chunk_text in enumerate(chunk_texts):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_count_for_section"] = chunk_count
            metadata["chunk_id"] = build_chunk_id(metadata, index)
            splits.append(Document(page_content=chunk_text, metadata=metadata))
    return splits
