from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CityRawFile:
    city_key: str
    display_name: str
    path: Path
    raw_text: str


@dataclass(frozen=True)
class CitySection:
    title: str
    level: int
    text: str
    include_in_rag: bool = True

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "text": self.text,
            "char_count": self.char_count,
        }


@dataclass
class CityArticle:
    schema_version: str
    city_key: str
    display_name: str
    source_path: Path
    raw_text: str
    clean_text: str
    metadata: dict[str, Any]
    sections: list[CitySection]
    warnings: list[str] = field(default_factory=list)

    def useful_sections(self) -> list[CitySection]:
        return [section for section in self.sections if section.include_in_rag]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "city_key": self.city_key,
            "display_name": self.display_name,
            "source": {
                "type": "wikipedia_text_dump",
                "path": str(self.source_path),
            },
            "metadata": self.metadata,
            "sections": [section.to_json() for section in self.sections],
            "quality": {
                "raw_char_count": len(self.raw_text),
                "clean_char_count": len(self.clean_text),
                "section_count": len(self.useful_sections()),
                "warnings": self.warnings,
            },
        }
