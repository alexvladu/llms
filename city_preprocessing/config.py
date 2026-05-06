from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1.0"
DEFAULT_COUNTRY = "România"

DEFAULT_INPUT_DIR = Path("informations/cities_text")
DEFAULT_JSON_OUTPUT_DIR = Path("informations/cities_json")
DEFAULT_CLEAN_TEXT_OUTPUT_DIR = Path("informations/cities_clean_text")
DEFAULT_REPORT_PATH = Path("informations/preprocessing_report.json")

NOISY_SECTION_TITLES = {
    "Note",
    "Bibliografie",
    "Vezi și",
    "Legături externe",
}

CANONICAL_SECTION_TITLES = {
    "geografie": "Geografie",
    "clima": "Climă",
    "climă": "Climă",
    "istorie": "Istorie",
    "demografie": "Demografie",
    "politica si administratie": "Politică și administrație",
    "politică și administrație": "Politică și administrație",
    "administratie": "Administrație",
    "administrație": "Administrație",
    "transport": "Transport",
    "economie": "Economie",
    "educatie": "Educație",
    "educație": "Educație",
    "sanatate": "Sănătate",
    "sănătate": "Sănătate",
    "cultura": "Cultură",
    "cultură": "Cultură",
    "turism": "Turism",
    "monumente istorice": "Monumente istorice",
    "obiective turistice": "Obiective turistice",
    "personalitati": "Personalități",
    "personalități": "Personalități",
    "note": "Note",
    "bibliografie": "Bibliografie",
    "vezi si": "Vezi și",
    "vezi și": "Vezi și",
    "legaturi externe": "Legături externe",
    "legături externe": "Legături externe",
}


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path = DEFAULT_INPUT_DIR
    json_output_dir: Path = DEFAULT_JSON_OUTPUT_DIR
    clean_text_output_dir: Path = DEFAULT_CLEAN_TEXT_OUTPUT_DIR
    report_path: Path = DEFAULT_REPORT_PATH
    include_noisy_sections: bool = False
    min_section_chars: int = 30
    write_clean_text: bool = True
    write_json: bool = True
    overwrite: bool = False
    verbose: bool = False
    noisy_section_titles: set[str] = field(default_factory=lambda: set(NOISY_SECTION_TITLES))
