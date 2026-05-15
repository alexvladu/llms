"""
city_loader.py
Reads all city JSON files produced by city_preprocessing and builds
a compact city_index.json used as the RAG context source during
synthetic training data generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Sections that are useful for relocation decisions
RELOCATION_SECTIONS = {
    "Climă",
    "Geografie",
    "Demografie",
    "Economie",
    "Transport",
    "Educație",
    "Sănătate",
    "Cultură",
    "Turism",
    "Obiective turistice",
    "Introducere",
}

SECTION_PREVIEW_CHARS = 400


def _preview(text: str) -> str:
    text = text.strip()
    if len(text) <= SECTION_PREVIEW_CHARS:
        return text
    # Try to cut at a sentence boundary
    cut = text[:SECTION_PREVIEW_CHARS]
    last_dot = cut.rfind(".")
    if last_dot > SECTION_PREVIEW_CHARS // 2:
        return cut[: last_dot + 1]
    return cut + "..."


def load_city_index(cities_json_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Build a compact city index from all *.json files in cities_json_dir.

    Returns a dict keyed by city_key. Each value is a flat dict containing
    metadata fields plus section_titles and section_previews.
    """
    cities_json_dir = Path(cities_json_dir)
    index: dict[str, dict[str, Any]] = {}

    for json_path in sorted(cities_json_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        city_key = data.get("city_key", json_path.stem)
        meta = data.get("metadata", {})

        entry: dict[str, Any] = {
            "city_key": city_key,
            "display_name": data.get("display_name", city_key),
            "county": meta.get("county"),
            "settlement_type": meta.get("settlement_type"),
            "population_2021": meta.get("population_2021"),
            "population_2011": meta.get("population_2011"),
            "has_hospital": meta.get("has_hospital"),
            "has_ambulance_station": meta.get("has_ambulance_station"),
            "has_polyclinic": meta.get("has_polyclinic"),
            "has_railway": meta.get("has_railway"),
            "has_national_roads": meta.get("has_national_roads"),
            "national_roads": meta.get("national_roads", []),
            "european_roads": meta.get("european_roads", []),
            "motorways": meta.get("motorways", []),
            "high_schools_count": meta.get("high_schools_count"),
            "schools_count": meta.get("schools_count"),
            "kindergartens_count": meta.get("kindergartens_count"),
            "education_institutions": meta.get("education_institutions", []),
            "ethnic_romanians_pct": meta.get("ethnic_romanians_pct"),
            "ethnic_roma_pct": meta.get("ethnic_roma_pct"),
            "religion_orthodox_pct": meta.get("religion_orthodox_pct"),
        }

        # Collect section titles and previews for relocation-relevant sections
        section_titles: list[str] = []
        section_previews: dict[str, str] = {}

        for section in data.get("sections", []):
            title = section.get("title", "")
            text = section.get("text", "")
            if title in RELOCATION_SECTIONS and text.strip():
                section_titles.append(title)
                section_previews[title] = _preview(text)

        entry["section_titles"] = section_titles
        entry["section_previews"] = section_previews

        index[city_key] = entry

    return index


def save_city_index(index: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_saved_city_index(index_path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(Path(index_path).read_text(encoding="utf-8"))


def build_city_card(
    city: dict[str, Any],
    include_sections: list[str] | None = None,
    max_sections: int = 3,
    max_section_chars: int = 150,
) -> str:
    """
    Render a compact text card for a city to inject into a [CONTEXT RAG] block.

    Parameters
    ----------
    include_sections   : whitelist of section titles (defaults to all relocation sections)
    max_sections       : cap on how many section previews to include (default 3)
    max_section_chars  : hard cap per section preview to keep output tokens manageable
    """
    lines: list[str] = []

    pop = city.get("population_2021") or city.get("population_2011")
    pop_str = f"{pop:,}".replace(",", ".") if pop else "necunoscută"
    county = city.get("county") or "necunoscut"
    stype = city.get("settlement_type") or "localitate"

    lines.append(f"{city['display_name']} (județul {county}, {stype}, {pop_str} locuitori):")

    # Infrastructure bullets (structured metadata — always short)
    infra: list[str] = []
    if city.get("has_railway"):
        infra.append("cale ferată")
    nr = city.get("national_roads") or []
    er = city.get("european_roads") or []
    mw = city.get("motorways") or []
    if nr:
        infra.append(f"DN ({', '.join(nr[:3])})")
    if er:
        infra.append(f"E ({', '.join(er[:3])})")
    if mw:
        infra.append(f"autostradă ({', '.join(mw[:2])})")
    if infra:
        lines.append(f"- Transport: {', '.join(infra)}")

    if city.get("has_hospital"):
        lines.append("- Sănătate: spital disponibil")

    edu = city.get("education_institutions") or []
    if edu:
        lines.append(f"- Educație: {', '.join(edu[:2])}")

    eth = city.get("ethnic_romanians_pct")
    if eth is not None:
        lines.append(f"- Demografie: {eth:.1f}% români")

    # Section previews — capped to avoid blowing the LLM output token budget
    previews = city.get("section_previews") or {}
    want = set(include_sections) if include_sections else set(RELOCATION_SECTIONS)
    priority = ["Climă", "Economie", "Transport", "Cultură", "Geografie", "Turism", "Educație", "Sănătate", "Demografie"]
    sections_added = 0
    for section in priority:
        if sections_added >= max_sections:
            break
        if section in want and section in previews:
            text = previews[section]
            if len(text) > max_section_chars:
                text = text[:max_section_chars].rsplit(" ", 1)[0] + "..."
            lines.append(f"- {section}: {text}")
            sections_added += 1

    return "\n".join(lines)
