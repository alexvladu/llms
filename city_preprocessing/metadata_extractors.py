from __future__ import annotations

import re
from typing import Any

from city_preprocessing.config import DEFAULT_COUNTRY
from city_preprocessing.models import CitySection

NUMBER_RE = r"\d{1,3}(?:\.\d{3})+|\d+"


def romanian_int(value: str) -> int:
    return int(value.replace(".", "").replace(" ", ""))


def romanian_pct(value: str) -> float:
    return float(value.replace(".", "").replace(",", "."))


def extract_county(text: str) -> str | None:
    match = re.search(r"\b(?:în|in|din)?\s*județul\s+([A-ZĂÂÎȘȚ][a-zăâîșțşţ-]+(?:\s+[A-ZĂÂÎȘȚ][a-zăâîșțşţ-]+)?)", text)
    if match:
        county = match.group(1).strip()
        county = re.split(r"[,.;\n]", county)[0].strip()
        return county
    return None


def extract_settlement_type(text: str) -> str:
    intro = text[:1200].lower()
    for settlement_type in ("municipiu", "oraș", "comună", "sat"):
        if re.search(rf"\b{settlement_type}\b", intro):
            return settlement_type
    return "unknown"


def extract_area(text: str) -> int | None:
    patterns = [
        rf"\bsuprafața\s+(?:municipiului|orașului|localității)?[^.\n]{{0,80}}\b({NUMBER_RE})\s*ha\b",
        rf"\b({NUMBER_RE})\s*ha\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = romanian_int(match.group(1))
            if 10 <= value <= 300000:
                return value
    return None


def extract_populations(text: str, warnings: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}

    for year in ("2021", "2011"):
        year_patterns = [
            rf"\b{year}\b[^.\n]{{0,180}}\b({NUMBER_RE})\s+de\s+locuitori\b",
            rf"\bpopulaț(?:ia|ie)[^.\n]{{0,180}}\b({NUMBER_RE})\s+de\s+locuitori[^.\n]{{0,80}}\({year}\)",
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            value = romanian_int(match.group(1))
            if 1 <= value <= 2_000_000:
                result[f"population_{year}"] = value
            else:
                warnings.append(f"suspicious_population_value:{year}:{value}")
            break

    if "population_2021" not in result:
        match = re.search(
            rf"populația\s+(?:municipiului|orașului|comunei|localității)[^.\n]{{0,120}}\bse\s+ridic[ăa]\s+la\s+({NUMBER_RE})\s+de\s+locuitori",
            text,
            re.I,
        )
        if match:
            value = romanian_int(match.group(1))
            if 1 <= value <= 2_000_000:
                result["population_2021"] = value

    if "population_2011" not in result:
        match = re.search(rf"când\s+fuseser[ăa]\s+înregistrați\s+({NUMBER_RE})\s+de\s+locuitori", text, re.I)
        if match:
            value = romanian_int(match.group(1))
            if 1 <= value <= 2_000_000:
                result["population_2011"] = value

    return result


PERCENTAGE_PATTERNS = {
    "ethnic_romanians_pct": r"Români\s*\(([\d,.]+)%\)",
    "ethnic_roma_pct": r"Romi\s*\(([\d,.]+)%\)",
    "religion_orthodox_pct": r"Ortodocși\s*\(([\d,.]+)%\)",
    "religion_pentecostal_pct": r"Penticostali\s*\(([\d,.]+)%\)",
    "religion_roman_catholic_pct": r"Romano-catolici\s*\(([\d,.]+)%\)",
}


def extract_percentages(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for field, pattern in PERCENTAGE_PATTERNS.items():
        match = re.search(pattern, text, re.I)
        if match:
            result[field] = romanian_pct(match.group(1))

    unknown_matches = re.findall(r"Necunoscut[ăa]\s*\(([\d,.]+)%\)", text, re.I)
    if unknown_matches:
        result["ethnicity_unknown_pct"] = romanian_pct(unknown_matches[0])
    if len(unknown_matches) > 1:
        result["religion_unknown_pct"] = romanian_pct(unknown_matches[1])
    return result


def section_text(sections: list[CitySection], title: str) -> str:
    return "\n\n".join(section.text for section in sections if section.title == title)


def extract_transport_metadata(text: str) -> dict[str, Any]:
    national_roads = sorted(set(re.findall(r"\bDN\d+[A-Z]?\b", text)))
    european_roads = sorted(set(re.findall(r"\bE\d{1,3}\b", text)))
    motorways = sorted(set(re.findall(r"\bA\d{1,2}\b", text)))
    rail_terms = r"\b(?:gară|gara|feroviar|tren|trenuri|cale ferată|căi ferate)\b"
    result: dict[str, Any] = {
        "has_national_roads": bool(national_roads),
        "has_railway": bool(re.search(rail_terms, text, re.I)),
    }
    if national_roads:
        result["national_roads"] = national_roads
    if european_roads:
        result["european_roads"] = european_roads
    if motorways:
        result["motorways"] = motorways
    return result


def extract_education_metadata(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    count_patterns = {
        "kindergartens_count": r"\b(\d+)\s+gr[ăa]diniț[ăe]\b",
        "schools_count": r"\b(\d+)\s+școli\b|\b(\d+)\s+scoli\b",
        "high_schools_count": r"\b(\d+)\s+licee\b",
    }
    for field, pattern in count_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            value = next(group for group in match.groups() if group is not None)
            result[field] = int(value)

    quoted = re.findall(r"[„\"]([^„”\"]{4,120})[”\"]", text)
    institutions: list[str] = []
    for name in quoted:
        context_start = max(0, text.find(name) - 80)
        context = text[context_start : text.find(name) + len(name) + 80]
        if re.search(r"colegi|lice|școal|scoal|gr[ăa]dini", context, re.I):
            full_name = name.strip()
            if full_name and full_name not in institutions:
                institutions.append(full_name)
    if institutions:
        result["education_institutions"] = institutions
    return result


def extract_healthcare_metadata(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if re.search(r"\bspital", text, re.I):
        result["has_hospital"] = True
    if re.search(r"\bambulanț[ăa]|\bambulanta", text, re.I):
        result["has_ambulance_station"] = True
    if re.search(r"\bpoliclinic", text, re.I):
        result["has_polyclinic"] = True
    return result


def extract_city_metadata(
    *,
    city_key: str,
    display_name: str,
    clean_text: str,
    sections: list[CitySection],
    warnings: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "city": display_name,
        "city_key": city_key,
        "country": DEFAULT_COUNTRY,
    }
    intro = section_text(sections, "Introducere") or clean_text[:1500]
    demography = section_text(sections, "Demografie")
    transport = section_text(sections, "Transport")
    education = section_text(sections, "Educație")
    healthcare = section_text(sections, "Sănătate")

    county = extract_county(intro)
    if county:
        metadata["county"] = county
    metadata["settlement_type"] = extract_settlement_type(intro)

    area = extract_area(clean_text)
    if area is not None:
        metadata["area_ha"] = area

    metadata.update(extract_populations(f"{intro}\n\n{demography}", warnings))
    metadata.update(extract_percentages(demography))
    metadata.update(extract_transport_metadata(transport))
    metadata.update(extract_education_metadata(education))
    metadata.update(extract_healthcare_metadata(healthcare))

    if len(metadata) <= 4:
        warnings.append("no_metadata_extracted")
    return metadata
