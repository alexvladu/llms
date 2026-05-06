from __future__ import annotations

import unittest
from pathlib import Path

from city_preprocessing.document_builder import build_city_documents
from city_preprocessing.metadata_extractors import (
    extract_city_metadata,
    extract_percentages,
    extract_populations,
    extract_transport_metadata,
)
from city_preprocessing.models import CityArticle, CitySection
from city_preprocessing.normalize import normalize_wikipedia_text
from city_preprocessing.sectioning import split_into_sections


class CityPreprocessingTests(unittest.TestCase):
    def test_normalization_preserves_paragraphs_and_collapses_noise(self) -> None:
        text = "Adjud\xa0 este   un municipiu\n\n\nGeografie"
        self.assertEqual(normalize_wikipedia_text(text), "Adjud este un municipiu\n\nGeografie")

    def test_sectioning_detects_romanian_headings_and_excludes_notes(self) -> None:
        text = """Adjud este un municipiu...
Geografie
Text geografie
Istorie
Text istorie
Note
↑ reference"""
        sections, warnings = split_into_sections(text)

        self.assertEqual([section.title for section in sections], ["Introducere", "Geografie", "Istorie"])
        self.assertIn("missing_demografie_section", warnings)

    def test_population_2021_extraction(self) -> None:
        warnings: list[str] = []
        metadata = extract_populations(
            "populația municipiului Adjud se ridică la 15.178 de locuitori",
            warnings,
        )
        self.assertEqual(metadata["population_2021"], 15178)

    def test_population_2011_extraction(self) -> None:
        warnings: list[str] = []
        metadata = extract_populations(
            "în scădere față de recensământul anterior din 2011, când fuseseră înregistrați 16.045 de locuitori",
            warnings,
        )
        self.assertEqual(metadata["population_2011"], 16045)

    def test_percentage_extraction(self) -> None:
        metadata = extract_percentages("Români (78,31%) Romi (5,57%)")
        self.assertEqual(
            metadata,
            {
                "ethnic_romanians_pct": 78.31,
                "ethnic_roma_pct": 5.57,
            },
        )

    def test_transport_extraction(self) -> None:
        metadata = extract_transport_metadata(
            "Orașul este traversat de DN2 și DN11A. Prin Gara Adjud trec trenuri zilnic."
        )
        self.assertEqual(metadata["national_roads"], ["DN11A", "DN2"])
        self.assertIs(metadata["has_national_roads"], True)
        self.assertIs(metadata["has_railway"], True)

    def test_document_building_adds_profile_and_section_headers(self) -> None:
        sections = [
            CitySection(
                title="Introducere",
                level=1,
                text="Adjud este un municipiu în județul Vrancea.",
            ),
            CitySection(
                title="Transport",
                level=1,
                text="Orașul este traversat de DN2 și are gară.",
            ),
        ]
        warnings: list[str] = []
        metadata = extract_city_metadata(
            city_key="adjud",
            display_name="Adjud",
            clean_text="\n".join(section.text for section in sections),
            sections=sections,
            warnings=warnings,
        )
        article = CityArticle(
            schema_version="1.0",
            city_key="adjud",
            display_name="Adjud",
            source_path=Path("cities_text/adjud.txt"),
            raw_text="raw",
            clean_text="clean",
            metadata=metadata,
            sections=sections,
            warnings=warnings,
        )

        documents = build_city_documents(article)

        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].metadata["section"], "Profil general")
        self.assertIn("Oraș: Adjud", documents[1].page_content)
        self.assertIn("Secțiune: Introducere", documents[1].page_content)
        self.assertEqual(documents[2].metadata["city_key"], "adjud")
        self.assertEqual(documents[2].metadata["section"], "Transport")


if __name__ == "__main__":
    unittest.main()
