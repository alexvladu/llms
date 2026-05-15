"""
conversation_validator.py
Quality checks applied to raw API-generated conversations before they
are added to the training dataset.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------
MIN_TURNS = 3          # user + assistant pairs (each counts as 1 turn)
MAX_TURNS = 7          # guard against runaway generations
MIN_TOTAL_CHARS = 300  # sanity check — something was actually generated
MAX_TOTAL_CHARS = 8000 # rough proxy for 2 048-token limit

# Romanian diacritics that should appear in genuine Romanian text
ROMANIAN_DIACRITICS = re.compile(r"[ăâîșțĂÂÎȘȚ]")

# A grounding phrase the assistant should use when presenting RAG context
GROUNDING_PHRASES = re.compile(
    r"(conform datelor|datele indic|din informaț|pe baza informaț|"
    r"potrivit datelor|conform informaț|recomand verificarea)",
    re.IGNORECASE,
)

# Clarification question marker
QUESTION_MARK = "?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    pass


def validate_conversation(raw: dict[str, Any], city_index: dict[str, Any]) -> list[str]:
    """
    Run all quality checks on a raw generated conversation dict.

    Returns a list of warning strings (empty = fully valid).
    Raises ValidationError for hard failures that disqualify the example.
    """
    warnings: list[str] = []

    conversation = raw.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValidationError("Missing or empty 'conversation' list")

    # Filter out system messages for turn counting
    dialogue_turns = [m for m in conversation if m.get("role") in ("user", "assistant")]

    # --- Turn count ---
    if len(dialogue_turns) < MIN_TURNS:
        raise ValidationError(
            f"Too few turns: {len(dialogue_turns)} (min {MIN_TURNS})"
        )
    if len(dialogue_turns) > MAX_TURNS:
        warnings.append(f"high_turn_count:{len(dialogue_turns)}")

    # --- Total length ---
    total_chars = sum(len(m.get("content", "")) for m in conversation)
    if total_chars < MIN_TOTAL_CHARS:
        raise ValidationError(f"Conversation too short: {total_chars} chars")
    if total_chars > MAX_TOTAL_CHARS:
        warnings.append("conversation_too_long")

    # --- Romanian language check ---
    all_text = " ".join(m.get("content", "") for m in dialogue_turns)
    if not ROMANIAN_DIACRITICS.search(all_text):
        raise ValidationError("No Romanian diacritics found — conversation may not be in Romanian")

    # --- Clarification check ---
    # At least one non-final assistant turn must contain a question mark
    assistant_turns = [m for m in dialogue_turns if m.get("role") == "assistant"]
    if len(assistant_turns) >= 2:
        clarification_turns = assistant_turns[:-1]  # all but final
        has_question = any(QUESTION_MARK in t.get("content", "") for t in clarification_turns)
        if not has_question:
            warnings.append("no_clarification_question")
    elif len(assistant_turns) == 1:
        # Single-turn response is acceptable but flag it
        warnings.append("single_assistant_turn")

    # --- Grounding check ---
    # Final assistant turn should cite context, not hallucinate
    if assistant_turns:
        final_response = assistant_turns[-1].get("content", "")
        if not GROUNDING_PHRASES.search(final_response):
            warnings.append("no_grounding_phrase")

    # --- Hallucination check ---
    # City names referenced in the final response must be present in [CONTEXT RAG]
    context_block = _extract_rag_context(conversation)
    if context_block and assistant_turns:
        final_response = assistant_turns[-1].get("content", "")
        unlisted_cities = _detect_unlisted_cities(final_response, context_block, city_index)
        if unlisted_cities:
            warnings.append(f"possible_hallucination:{','.join(unlisted_cities[:3])}")

    # --- Metadata check ---
    meta = raw.get("metadata", {})
    if not meta.get("recommendation_given"):
        warnings.append("no_recommendation_flag")

    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_rag_context(conversation: list[dict]) -> str:
    """Extract the text inside [CONTEXT RAG]...[END CONTEXT RAG] system messages."""
    for msg in conversation:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if "[CONTEXT RAG]" in content:
                return content
    return ""


def _detect_unlisted_cities(
    response_text: str,
    context_block: str,
    city_index: dict[str, Any],
) -> list[str]:
    """
    Return display names of cities mentioned in response_text that do NOT
    appear anywhere in context_block (loose substring match).
    """
    unlisted: list[str] = []
    response_lower = response_text.lower()
    context_lower = context_block.lower()

    for city_key, city in city_index.items():
        display = city.get("display_name", city_key)
        display_lower = display.lower()
        # City appears in response but not in context → possible hallucination
        if display_lower in response_lower and display_lower not in context_lower:
            unlisted.append(display)

    return unlisted
