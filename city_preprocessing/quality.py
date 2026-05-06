from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from city_preprocessing.models import CityArticle


@dataclass
class CityProcessingResult:
    city: str
    city_key: str
    status: str
    raw_length: int = 0
    clean_length: int = 0
    sections_found: list[str] = field(default_factory=list)
    metadata_fields_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    path: str | None = None
    json_path: str | None = None
    clean_text_path: str | None = None
    document_count: int = 0

    @classmethod
    def from_article(
        cls,
        article: CityArticle,
        *,
        json_path: str | None = None,
        clean_text_path: str | None = None,
        document_count: int = 0,
    ) -> "CityProcessingResult":
        return cls(
            city=article.display_name,
            city_key=article.city_key,
            status="ok",
            raw_length=len(article.raw_text),
            clean_length=len(article.clean_text),
            sections_found=[section.title for section in article.useful_sections()],
            metadata_fields_found=sorted(article.metadata.keys()),
            warnings=list(article.warnings),
            path=str(article.source_path),
            json_path=json_path,
            clean_text_path=clean_text_path,
            document_count=document_count,
        )

    def to_json(self) -> dict[str, Any]:
        data = {
            "city": self.city,
            "city_key": self.city_key,
            "status": self.status,
            "raw_length": self.raw_length,
            "clean_length": self.clean_length,
            "sections_found": self.sections_found,
            "metadata_fields_found": self.metadata_fields_found,
            "warnings": self.warnings,
            "document_count": self.document_count,
        }
        if self.path:
            data["path"] = self.path
        if self.json_path:
            data["json_path"] = self.json_path
        if self.clean_text_path:
            data["clean_text_path"] = self.clean_text_path
        if self.error:
            data["error"] = self.error
        return data


def build_global_report(results: list[CityProcessingResult], *, total_files: int) -> dict[str, Any]:
    ok_results = [result for result in results if result.status == "ok"]
    failed_results = [result for result in results if result.status != "ok"]
    total_sections = sum(len(result.sections_found) for result in ok_results)
    warning_items = [
        {"city": result.city_key or result.city, "warning": warning}
        for result in results
        for warning in result.warnings
    ]
    metadata_counter = Counter(
        field
        for result in ok_results
        for field in result.metadata_fields_found
    )
    created_rag_documents = sum(result.document_count for result in ok_results)
    return {
        "total_files": total_files,
        "processed_ok": len(ok_results),
        "failed": len(failed_results),
        "total_sections": total_sections,
        "avg_sections_per_city": round(total_sections / len(ok_results), 2) if ok_results else 0,
        "created_rag_documents": created_rag_documents,
        "created_section_documents": created_rag_documents,
        "cities_with_population_2021": metadata_counter.get("population_2021", 0),
        "cities_with_county": metadata_counter.get("county", 0),
        "warnings": warning_items,
        "errors": [
            {
                "city": result.city_key or result.city or "unknown",
                "path": result.path,
                "error": result.error,
            }
            for result in failed_results
        ],
        "cities": [result.to_json() for result in results],
    }
