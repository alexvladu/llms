"""
api_client.py
Thin wrappers around Groq, Gemini, Anthropic, and OpenAI APIs for
synthetic training data generation. Handles retries with exponential
backoff and a polite sleep between requests to avoid rate limits.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------
DEFAULT_MODEL_GROQ      = "llama-3.3-70b-versatile"
DEFAULT_MODEL_GEMINI    = "gemini-2.0-flash"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_MODEL_OPENAI    = "gpt-4o-mini"

MAX_TOKENS   = 4096
TEMPERATURE  = 0.9   # Higher temp → more varied conversations
MAX_RETRIES  = 3
BACKOFF_BASE = 2.0   # Seconds; doubles on each retry

# Groq free tier (llama-3.3-70b-versatile): 30 RPM, ~6000 TPM.
# 12s sleep ≈ 5 req/min which stays safely under both caps.
INTER_REQUEST_SLEEP_GROQ    = 12.0
INTER_REQUEST_SLEEP_GEMINI  = 4.0
INTER_REQUEST_SLEEP_DEFAULT = 1.2


def _exponential_backoff(attempt: int) -> None:
    wait = BACKOFF_BASE * (2 ** attempt)
    print(f"  Rate limit / error — waiting {wait:.0f}s before retry {attempt + 1}/{MAX_RETRIES}")
    time.sleep(wait)


def _strip_code_fence(text: str) -> str:
    """Remove markdown ```json ... ``` fences that models sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _sanitize_json_text(text: str) -> str:
    """
    Fix common LLM JSON quirks before parsing:
    - Normalize line endings
    - Replace literal tab characters with the JSON escape sequence
    Does NOT fix structurally broken JSON (e.g. actual truncation).
    """
    import re
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\t', r'\\t', text)
    return text


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    raw_text = _strip_code_fence(raw_text)
    raw_text = _sanitize_json_text(raw_text)
    return json.loads(raw_text)


# ---------------------------------------------------------------------------
# Groq client  (primary / recommended — free tier, Llama 3.3 70B)
# ---------------------------------------------------------------------------

class GroqClient:
    """
    Generates one conversation JSON object per call using the Groq API.
    Groq is OpenAI-compatible, so this reuses the openai package.

    Install:  pip install openai          (openai package, NOT groq-specific)
    API key:  https://console.groq.com/keys → set GROQ_API_KEY=gsk_...

    Default model : llama-3.3-70b-versatile  (best quality on the free tier)
    Faster option : llama-3.1-8b-instant     (higher TPM allowance)

    Free-tier limits (llama-3.3-70b-versatile):
        30 req/min, ~6 000 TPM  →  sleep 12s between calls (≈5 req/min)
    """

    inter_request_sleep = INTER_REQUEST_SLEEP_GROQ

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL_GROQ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install 'openai' package: pip install openai") from exc

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key required. Get one free at https://console.groq.com/keys "
                "then set GROQ_API_KEY env var or pass api_key=."
            )
        self._client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
        )
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import openai

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                raw_text = response.choices[0].message.content.strip()
                finish = response.choices[0].finish_reason
                if finish == "length":
                    raise RuntimeError(
                        f"Groq hit MAX_TOKENS limit ({MAX_TOKENS}). "
                        "Increase MAX_TOKENS in api_client.py or reduce city card size."
                    )
                return _parse_json_response(raw_text)

            except openai.RateLimitError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except openai.APIError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except json.JSONDecodeError as exc:
                if attempt < 1:
                    print(f"  JSON parse error — retrying once...")
                    last_error = exc
                else:
                    raise RuntimeError(
                        f"Groq returned malformed JSON. "
                        f"Raw tail: ...{getattr(exc, 'doc', '')[-200:]!r}"
                    ) from exc

        raise RuntimeError(f"All {MAX_RETRIES} retries failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Generates one conversation JSON object per call using the Google Gemini API.

    Install:  pip install google-generativeai
    API key:  set GEMINI_API_KEY=...  (or pass api_key= directly)
    """

    inter_request_sleep = INTER_REQUEST_SLEEP_GEMINI

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL_GEMINI) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Install the Gemini SDK:  pip install google-generativeai"
            ) from exc

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key=."
            )
        genai.configure(api_key=key)
        self.model = model
        self._genai = genai

    def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                gemini_model = self._genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt,
                    generation_config=self._genai.GenerationConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS,
                        response_mime_type="application/json",
                    ),
                )
                response = gemini_model.generate_content(user_prompt)

                candidate = response.candidates[0] if response.candidates else None
                if candidate is not None:
                    reason = str(candidate.finish_reason)
                    if "MAX_TOKENS" in reason or reason == "2":
                        raise RuntimeError(
                            f"Gemini hit MAX_TOKENS limit ({MAX_TOKENS}). "
                            "Increase MAX_TOKENS in api_client.py."
                        )
                    if "SAFETY" in reason or reason == "3":
                        raise RuntimeError(
                            f"Gemini blocked by safety filters (finish_reason={reason})."
                        )

                raw_text = response.text.strip()
                return _parse_json_response(raw_text)

            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    _exponential_backoff(attempt)
                    last_error = exc
                elif isinstance(exc, json.JSONDecodeError):
                    if attempt < 1:
                        print(f"  JSON parse error — retrying once...")
                        last_error = exc
                    else:
                        raise RuntimeError(
                            f"Gemini returned malformed JSON. "
                            f"Raw tail: ...{getattr(exc, 'doc', '')[-200:]!r}"
                        ) from exc
                else:
                    raise RuntimeError(f"Gemini API error: {exc}") from exc

        raise RuntimeError(f"All {MAX_RETRIES} retries failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient:
    """Generates one conversation JSON object per call using the Claude API."""

    inter_request_sleep = INTER_REQUEST_SLEEP_DEFAULT

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL_ANTHROPIC) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install 'anthropic' package: pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import anthropic

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw_text = response.content[0].text
                return _parse_json_response(raw_text)
            except anthropic.RateLimitError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except anthropic.APIError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except json.JSONDecodeError as exc:
                _exponential_backoff(attempt)
                last_error = exc

        raise RuntimeError(f"All {MAX_RETRIES} retries failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class OpenAIClient:
    """Generates one conversation JSON object per call using the OpenAI API."""

    inter_request_sleep = INTER_REQUEST_SLEEP_DEFAULT

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL_OPENAI) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install 'openai' package: pip install openai") from exc

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import openai

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                raw_text = response.choices[0].message.content.strip()
                return _parse_json_response(raw_text)
            except openai.RateLimitError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except openai.APIError as exc:
                _exponential_backoff(attempt)
                last_error = exc
            except json.JSONDecodeError as exc:
                _exponential_backoff(attempt)
                last_error = exc

        raise RuntimeError(f"All {MAX_RETRIES} retries failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_client(backend: str = "groq", api_key: str | None = None, model: str | None = None):
    """
    Factory that returns the appropriate API client.

    Parameters
    ----------
    backend : "groq" (default) | "gemini" | "anthropic" | "openai"
    api_key : optional; falls back to env vars:
                GROQ_API_KEY                     (for groq)
                GEMINI_API_KEY / GOOGLE_API_KEY  (for gemini)
                ANTHROPIC_API_KEY                (for anthropic)
                OPENAI_API_KEY                   (for openai)
    model   : optional override for the default model per backend
    """
    if backend == "groq":
        return GroqClient(api_key=api_key, model=model or DEFAULT_MODEL_GROQ)
    if backend == "gemini":
        return GeminiClient(api_key=api_key, model=model or DEFAULT_MODEL_GEMINI)
    if backend == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or DEFAULT_MODEL_ANTHROPIC)
    if backend == "openai":
        return OpenAIClient(api_key=api_key, model=model or DEFAULT_MODEL_OPENAI)
    raise ValueError(f"Unknown backend '{backend}'. Use 'groq', 'gemini', 'anthropic', or 'openai'.")
