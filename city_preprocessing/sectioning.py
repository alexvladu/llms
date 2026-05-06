from __future__ import annotations

import re
import unicodedata

from city_preprocessing.config import CANONICAL_SECTION_TITLES, NOISY_SECTION_TITLES
from city_preprocessing.models import CitySection


def normalize_lookup_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = value.replace("ș", "s").replace("ş", "s").replace("ț", "t").replace("ţ", "t")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


CANONICAL_BY_LOOKUP = {
    normalize_lookup_key(raw_title): canonical_title
    for raw_title, canonical_title in CANONICAL_SECTION_TITLES.items()
}


def canonical_section_title(line: str) -> str | None:
    stripped = line.strip().strip(":")
    if len(stripped) > 60:
        return None
    if re.search(r"[.!?]", stripped):
        return None
    return CANONICAL_BY_LOOKUP.get(normalize_lookup_key(stripped))


def has_structured_fact(text: str) -> bool:
    return bool(re.search(r"\b(?:DN\d+[A-Z]?|E\d+|A\d+|\d[\d.]*\s*(?:ha|locuitori)|\d+(?:,\d+)?%)\b", text, re.I))


def split_into_sections(
    text: str,
    *,
    include_noisy_sections: bool = False,
    min_section_chars: int = 0,
) -> tuple[list[CitySection], list[str]]:
    sections: list[CitySection] = []
    warnings: list[str] = []
    current_title = "Introducere"
    current_lines: list[str] = []
    seen_titles: set[str] = set()
    found_heading = False

    def flush() -> None:
        nonlocal current_lines
        section_text = "\n".join(line for line in current_lines).strip()
        current_lines = []
        if not section_text:
            if current_title == "Introducere":
                warnings.append("missing_intro")
            return
        include_in_rag = current_title not in NOISY_SECTION_TITLES
        if include_in_rag and len(section_text) < min_section_chars and not has_structured_fact(section_text):
            return
        if not include_in_rag and not include_noisy_sections:
            return
        sections.append(
            CitySection(
                title=current_title,
                level=1,
                text=section_text,
                include_in_rag=include_in_rag,
            )
        )

    for line in text.splitlines():
        heading = canonical_section_title(line)
        if heading:
            flush()
            if heading in seen_titles:
                warnings.append(f"duplicate_section_title:{heading}")
            seen_titles.add(heading)
            current_title = heading
            current_lines = []
            found_heading = True
            continue
        current_lines.append(line)

    flush()
    if not found_heading:
        warnings.append("large_unsectioned_text" if len(text) > 2000 else "no_sections_detected")
    if not any(section.title == "Demografie" for section in sections):
        warnings.append("missing_demografie_section")
    return sections, warnings
