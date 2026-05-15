"""
dataset_formatter.py
Converts raw API-generated conversations into the ChatML messages format
expected by Qwen's tokenizer and TRL's SFTTrainer.

Training data design
--------------------
Each example contains:
  - system  : the fixed Romanian city-advisor system prompt
  - user    : initial relocation request (turn 1)
  - assistant: clarifying questions (turn 2)
  - ... (additional preference-elicitation turns) ...
  - system  : [CONTEXT RAG] block with city data injected by the application
  - assistant: final grounded recommendation citing only the RAG context

The model is trained with loss only on assistant turns, so it learns:
  1. How to ask good clarifying questions (no city facts needed)
  2. How to present RAG-returned city data without adding memorised facts
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "Ești un asistent specializat în recomandarea orașelor din România pentru relocare. "
    "Rolul tău are două etape:\n"
    "1. Colectezi preferințele utilizatorului prin întrebări clare despre climă, economie, "
    "transport, educație, sănătate, cultură și cost al vieții.\n"
    "2. Când ai suficiente informații, prezinți EXCLUSIV datele furnizate în blocul "
    "[CONTEXT RAG]. Nu adăuga fapte despre orașe din propria cunoaștere."
)


def raw_to_messages(raw: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert a raw generated conversation dict to a ChatML messages list.

    The raw conversation already contains a [CONTEXT RAG] system message
    injected just before the final assistant recommendation turn.
    We prepend the fixed system prompt and return the full messages list.
    """
    conversation: list[dict[str, str]] = raw.get("conversation", [])

    # Prepend the fixed system prompt as the very first message
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in conversation)
    return messages


def format_training_example(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a single training record ready for JSONL export."""
    return {
        "messages": raw_to_messages(raw),
        "metadata": raw.get("metadata", {}),
    }


def load_raw_conversations(raw_file: Path) -> list[dict[str, Any]]:
    """Load all conversations from a JSONL file."""
    raw_file = Path(raw_file)
    conversations: list[dict[str, Any]] = []
    for line in raw_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return conversations


def split_and_save(
    conversations: list[dict[str, Any]],
    output_dir: Path,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Shuffle, split into train/val, format, and write JSONL files.

    Returns (train_path, val_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    data = list(conversations)
    rng.shuffle(data)

    split_idx = max(1, int(len(data) * (1 - val_fraction)))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_path = output_dir / "dataset_train.jsonl"
    val_path = output_dir / "dataset_val.jsonl"

    _write_jsonl(train_path, train_data)
    _write_jsonl(val_path, val_data)

    print(f"Train: {len(train_data)} examples → {train_path}")
    print(f"Val:   {len(val_data)} examples → {val_path}")

    return train_path, val_path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(format_training_example(record), ensure_ascii=False) + "\n")
