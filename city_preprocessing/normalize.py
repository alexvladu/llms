from __future__ import annotations

import re

REFERENCE_NOISE_RE = re.compile(
    r"""(?ix)
    ^
    (?:
        ↑.*
        |\[\s*\d+\s*\]
        |\d+\.\s*$
        |date:\s*recens[aă]minte.*
        |adjud\s*-\s*evoluția\s*demografică
        |populația\s+istorică\s+din\s+.+$
    )
    $
    """
)


def is_reference_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if REFERENCE_NOISE_RE.match(stripped):
        return True
    if stripped.count("↑") >= 2:
        return True
    return False


def normalize_wikipedia_text(text: str) -> str:
    """Clean flattened Wikipedia text while preserving Romanian content."""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if is_reference_noise_line(line):
            continue
        line = re.sub(r"[ \t\f\v]+", " ", line)
        cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized
