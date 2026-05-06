from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from city_preprocessing.config import (
    DEFAULT_CLEAN_TEXT_OUTPUT_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_JSON_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    PipelineConfig,
    SCHEMA_VERSION,
)
from city_preprocessing.document_builder import build_city_documents
from city_preprocessing.metadata_extractors import extract_city_metadata
from city_preprocessing.models import CityArticle, CityRawFile
from city_preprocessing.normalize import normalize_wikipedia_text
from city_preprocessing.quality import CityProcessingResult, build_global_report
from city_preprocessing.sectioning import split_into_sections


def display_name_from_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", "-").title()


def discover_city_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.txt"), key=lambda path: path.name)


def read_city_file(path: Path) -> CityRawFile:
    raw_text = path.read_text(encoding="utf-8")
    first_line = next((line.strip() for line in raw_text.splitlines() if line.strip()), "")
    display_name = first_line if first_line and len(first_line) <= 80 else display_name_from_stem(path.stem)
    return CityRawFile(
        city_key=path.stem,
        display_name=display_name,
        path=path,
        raw_text=raw_text,
    )


def validate_raw_file(raw_file: CityRawFile) -> list[str]:
    warnings: list[str] = []
    stripped = raw_file.raw_text.strip()
    if not stripped:
        warnings.append("empty_file")
    if len(stripped) < 100:
        warnings.append("very_short_file")
    nonempty_lines = [line for line in stripped.splitlines() if line.strip()]
    if len(nonempty_lines) <= 1:
        warnings.append("only_title")
    return warnings


def process_raw_city(
    raw_file: CityRawFile,
    *,
    include_noisy_sections: bool = False,
    min_section_chars: int = 30,
) -> CityArticle:
    warnings = validate_raw_file(raw_file)
    clean_text = normalize_wikipedia_text(raw_file.raw_text)
    sections, section_warnings = split_into_sections(
        clean_text,
        include_noisy_sections=include_noisy_sections,
        min_section_chars=min_section_chars,
    )
    warnings.extend(section_warnings)
    metadata = extract_city_metadata(
        city_key=raw_file.city_key,
        display_name=raw_file.display_name,
        clean_text=clean_text,
        sections=sections,
        warnings=warnings,
    )
    return CityArticle(
        schema_version=SCHEMA_VERSION,
        city_key=raw_file.city_key,
        display_name=raw_file.display_name,
        source_path=raw_file.path,
        raw_text=raw_file.raw_text,
        clean_text=clean_text,
        metadata=metadata,
        sections=sections,
        warnings=sorted(set(warnings)),
    )


def process_city_file(
    path: Path,
    *,
    include_noisy_sections: bool = False,
    min_section_chars: int = 30,
) -> CityArticle:
    return process_raw_city(
        read_city_file(path),
        include_noisy_sections=include_noisy_sections,
        min_section_chars=min_section_chars,
    )


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    files = discover_city_files(input_dir)
    results: list[CityProcessingResult] = []
    seen_display_names: set[str] = set()

    for path in files:
        try:
            raw_file = read_city_file(path)
            duplicate_warnings = []
            if raw_file.display_name in seen_display_names:
                duplicate_warnings.append("duplicate_city_name")
            seen_display_names.add(raw_file.display_name)
            article = process_raw_city(
                raw_file,
                include_noisy_sections=config.include_noisy_sections,
                min_section_chars=config.min_section_chars,
            )
            article.warnings.extend(w for w in duplicate_warnings if w not in article.warnings)
            documents = build_city_documents(article)

            json_path = config.json_output_dir / f"{article.city_key}.json"
            clean_text_path = config.clean_text_output_dir / f"{article.city_key}.txt"
            if config.write_json:
                write_json(json_path, article.to_json(), overwrite=config.overwrite)
            if config.write_clean_text:
                write_text(clean_text_path, article.clean_text, overwrite=config.overwrite)
            results.append(
                CityProcessingResult.from_article(
                    article,
                    json_path=str(json_path) if config.write_json else None,
                    clean_text_path=str(clean_text_path) if config.write_clean_text else None,
                    document_count=len(documents),
                )
            )
            if config.verbose:
                print(f"Processed {article.city_key}: {len(article.useful_sections())} sections")
        except Exception as exc:  # noqa: BLE001 - batch processing must continue.
            results.append(
                CityProcessingResult(
                    city=path.stem,
                    city_key=path.stem,
                    status="failed",
                    path=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if config.verbose:
                print(f"Failed {path}: {exc}", file=sys.stderr)

    report = build_global_report(results, total_files=len(files))
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_city_documents(input_dir: Path) -> list[Any]:
    documents: list[Any] = []
    for path in discover_city_files(Path(input_dir)):
        article = process_city_file(path)
        documents.extend(build_city_documents(article))
    return documents


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Romanian Wikipedia-derived city text files for RAG.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--json-output-dir", type=Path, default=DEFAULT_JSON_OUTPUT_DIR)
    parser.add_argument("--clean-text-output-dir", type=Path, default=DEFAULT_CLEAN_TEXT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-noisy-sections", action="store_true")
    parser.add_argument("--min-section-chars", type=int, default=30)
    parser.add_argument("--write-clean-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = discover_city_files(args.input_dir)
    print(f"Found {len(files)} files")
    report = run_pipeline(
        PipelineConfig(
            input_dir=args.input_dir,
            json_output_dir=args.json_output_dir,
            clean_text_output_dir=args.clean_text_output_dir,
            report_path=args.report_path,
            include_noisy_sections=args.include_noisy_sections,
            min_section_chars=args.min_section_chars,
            write_clean_text=args.write_clean_text,
            write_json=args.write_json,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    )
    json_count = len(list(args.json_output_dir.glob("*.json"))) if args.write_json and args.json_output_dir.exists() else 0
    clean_count = len(list(args.clean_text_output_dir.glob("*.txt"))) if args.write_clean_text and args.clean_text_output_dir.exists() else 0
    print(f"Processed {report['processed_ok'] + report['failed']} files")
    print(f"Created {json_count} JSON files")
    print(f"Created {clean_count} clean text files")
    print(f"Created {report['created_section_documents']} RAG documents")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Report written to {args.report_path}")
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
