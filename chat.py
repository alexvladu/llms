#!/usr/bin/env python3
"""
chat.py — Romanian city relocation advisor (pre-fine-tuning version)

Two-phase conversational RAG pipeline:
  Phase 1 — Qwen asks clarifying questions to gather relocation preferences.
             No city facts are mentioned.
  Phase 2 — After enough preferences are collected, ChromaDB is queried and
             the results are injected as a [CONTEXT RAG] system message.
             Qwen then presents ONLY what is in that block.

Usage:
    python chat.py

Prerequisites:
    LM Studio running on http://localhost:1234 with:
      - Qwen 2.5-3B-Instruct (or any Qwen variant) loaded as the chat model
      - text-embedding-nomic-embed-text-v1.5 loaded as the embedding model
    ChromaDB populated (run main.ipynb cells 1-5 first if chroma_cities/ is missing).
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import chromadb
import requests
from openai import OpenAI
from flask import Flask, request, jsonify
import uuid
from threading import Lock
import traceback


def _init_resources():
    """Initialize ChromaDB collection and LLM client; return (llm_client, collection, city_map)."""
    if not CHROMA_DIR.exists():
        print(
            f"EROARE: ChromaDB nu a fost găsit la «{CHROMA_DIR}».\n"
            "Rulați mai întâi celulele 1-5 din main.ipynb pentru a construi baza de date."
        )
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"EROARE: Nu s-a putut deschide colecția «{COLLECTION_NAME}»: {exc}")
        sys.exit(1)

    city_map = _load_city_map(collection)
    print(f"✓ ChromaDB: {collection.count()} fragmente, {len(city_map)} orașe")

    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

    try:
        test_vec = get_embedding("test de conexiune")
        print(f"✓ Embedding model: {len(test_vec)} dimensiuni")
    except Exception as exc:
        print(f"EROARE la embedding: {exc}\nVerificați că LM Studio rulează pe {LM_STUDIO_URL}")
        sys.exit(1)

    try:
        llm_client.models.list()
        print("✓ LLM conectat")
    except Exception as exc:
        print(f"AVERTISMENT: Nu s-a putut verifica LLM-ul ({exc}). Continuăm oricum...")

    return llm_client, collection, city_map

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

CHROMA_DIR = REPO_ROOT / "chroma_cities"
COLLECTION_NAME = "romanian_cities"

LM_STUDIO_URL = "http://localhost:1234/v1"
CHAT_MODEL    = "qwen"          # LM Studio serves the loaded model under any name
EMBED_MODEL   = "text-embedding-nomic-embed-text-v1.5"

RAG_TRIGGER_TURNS   = 3   # fallback: always trigger RAG after this many user turns
RAG_TRIGGER_WORDS   = 70  # trigger early if user has typed this many words total
                          # (~2 detailed sentences covers most preference dimensions)
RAG_TOP_K           = 6   # chunks to retrieve per query
MAX_RESPONSE_TOKENS = 700

RESPONSE_LIMIT_PROMPT = (
    "Răspunde în limba română, direct și concis, în maximum 150 de cuvinte. "
    "Nu include raționament intern, pași de analiză sau explicații despre cum gândești."
)

# Reuse the exact system prompt from dataset_formatter so inference behaviour
# mirrors what the fine-tuned model will see.
from fine_tuning.src.dataset_formatter import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Embedding  (direct HTTP — no LangChain needed)
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """POST to LM Studio's /v1/embeddings and return the vector."""
    resp = requests.post(
        f"{LM_STUDIO_URL}/embeddings",
        json={"input": [text], "model": EMBED_MODEL},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# ChromaDB retrieval helpers  (mirrors query_city_chroma.py logic)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_city_map(collection) -> dict[str, str]:
    """Return {city_key: display_name} for every chunk in the collection."""
    data = collection.get(include=["metadatas"], limit=100_000)
    return {
        m["city_key"]: m.get("city", m["city_key"])
        for m in data["metadatas"]
        if m and "city_key" in m
    }


def _detect_city(query: str, city_map: dict[str, str]) -> str | None:
    norm_q = f" {_normalize(query)} "
    aliases: dict[str, set[str]] = {}
    for key, name in city_map.items():
        for variant in (_normalize(name), _normalize(key)):
            aliases.setdefault(variant, set()).add(key)
        parts = _normalize(name).split()
        if len(parts) > 1 and len(parts[0]) >= 4:
            aliases.setdefault(parts[0], set()).add(key)
    matches: set[str] = set()
    for alias, keys in aliases.items():
        if f" {alias} " in norm_q:
            matches.update(keys)
    return next(iter(matches)) if len(matches) == 1 else None


def _detect_section(query: str) -> str | None:
    norm_q = _normalize(query)
    for section, pattern in [
        ("Demografie", r"\b(populatie|populatia|locuitori|demografie)\b"),
        ("Transport",  r"\b(tren|gara|transport|feroviar|autostrada)\b"),
        ("Educație",   r"\b(scoala|liceu|universitate|educatie)\b"),
        ("Sănătate",   r"\b(spital|sanatate|policlinica)\b"),
        ("Geografie",  r"\b(clima|relief|munte|rau|geografie)\b"),
        ("Economie",   r"\b(job|locuri de munca|industrie|economie|it|salari)\b"),
    ]:
        if re.search(pattern, norm_q):
            return section
    return None


def retrieve_chunks(
    query: str,
    collection,
    city_map: dict[str, str],
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant city chunks for a free-text query.
    Uses the pre-computed nomic embedding to query chromadb directly,
    so no embedding function needs to be attached to the collection.
    """
    vector = get_embedding(query)

    # Build optional metadata filter
    city_key  = _detect_city(query, city_map)
    section   = _detect_section(query)

    where: dict | None = None
    if city_key and section:
        where = {"$and": [{"city_key": city_key}, {"section": section}]}
    elif city_key:
        where = {"city_key": city_key}
    elif section:
        where = {"section": section}

    result = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({"text": doc.strip(), "meta": meta, "distance": dist})
    return chunks


# ---------------------------------------------------------------------------
# [CONTEXT RAG] block builder
# ---------------------------------------------------------------------------

def build_rag_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the [CONTEXT RAG] ... [END CONTEXT RAG] block
    that matches the training data format.
    """
    # Group by city, preserving order of first appearance
    city_sections: dict[str, list[str]] = {}
    city_meta: dict[str, dict] = {}

    for chunk in chunks:
        meta   = chunk["meta"]
        city   = meta.get("city", "Necunoscut")
        section = meta.get("section", "")
        text   = chunk["text"]

        if city not in city_sections:
            city_sections[city] = []
            city_meta[city] = meta

        city_sections[city].append(f"  [{section}]: {text[:400]}")

    lines = ["[CONTEXT RAG]"]
    for city, sections in city_sections.items():
        m = city_meta[city]
        county  = m.get("county", "")
        stype   = m.get("settlement_type", "")
        pop     = m.get("population_2021") or m.get("population_2011") or ""
        header  = city
        if county or stype:
            header += f" ({', '.join(filter(None, [county, stype]))})"
        if pop:
            header += f" — {pop:,} locuitori".replace(",", ".")
        lines.append(f"\n{header}:")
        lines.extend(sections)

    lines.append("\n[END CONTEXT RAG]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def llm_chat(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int = MAX_RESPONSE_TOKENS,
    temperature: float = 0.7,
) -> str:
    constrained_messages = [{"role": "system", "content": RESPONSE_LIMIT_PROMPT}]
    constrained_messages.extend(messages)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=constrained_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def extract_search_query(conversation: list[dict], client: OpenAI) -> str:
    """
    Ask the LLM to distil a ChromaDB search query from the conversation so far.
    Returns a short Romanian query string capturing the user's relocation preferences.
    """
    summary_prompt = (
        "Pe baza conversației de mai jos, scrie o singură propoziție de căutare în română "
        "care să descrie ce tip de oraș din România caută utilizatorul. "
        "Include preferințele menționate despre climă, locuri de muncă, transport, "
        "dimensiunea orașului, cost al vieții, natură și cultură. "
        "Returnează DOAR propoziția de căutare, fără altceva."
    )
    extract_messages = [
        {"role": "system", "content": summary_prompt},
        {
            "role": "user",
            "content": "Conversație:\n"
            + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
            + "\n\nQuery de căutare:",
        },
    ]
    try:
        return llm_chat(client, extract_messages, max_tokens=120, temperature=0.2)
    except Exception:
        # Fallback: concatenate the last two user messages
        user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
        return " ".join(user_msgs[-2:])


# ---------------------------------------------------------------------------
# Conversational agent
# ---------------------------------------------------------------------------

class CityAdvisor:
    """
    Stateful two-phase conversational agent for Romanian city relocation advice.

    Phase 1 (turns < RAG_TRIGGER_TURNS):
        Qwen asks clarifying questions. No city facts injected.

    Phase 2 (turns >= RAG_TRIGGER_TURNS):
        ChromaDB queried once; results injected as [CONTEXT RAG] system message.
        On subsequent turns the stored context stays in the message history so
        follow-up questions remain grounded.
    """

    def __init__(self, client: OpenAI, collection, city_map: dict[str, str]) -> None:
        self.client     = client
        self.collection = collection
        self.city_map   = city_map
        self.history: list[dict] = []   # user / assistant turns only
        self.rag_block: str | None = None
        self.rag_done  = False

    # ------------------------------------------------------------------ #
    def _user_turn_count(self) -> int:
        return sum(1 for m in self.history if m["role"] == "user")

    def _should_trigger_rag(self) -> bool:
        if self.rag_done or self._user_turn_count() == 0:
            return False
        # Always trigger after N turns of back-and-forth
        if self._user_turn_count() >= RAG_TRIGGER_TURNS:
            return True
        # Trigger early when the user has given enough detail upfront
        # (a long first message covers most preference dimensions already)
        total_words = sum(
            len(m["content"].split())
            for m in self.history
            if m["role"] == "user"
        )
        return total_words >= RAG_TRIGGER_WORDS

    def _do_rag(self) -> None:
        search_query = extract_search_query(self.history, self.client)
        print(f"\n  [RAG] Căutare: «{search_query}»")
        chunks = retrieve_chunks(search_query, self.collection, self.city_map)
        self.rag_block = build_rag_block(chunks)
        self.rag_done  = True
        cities_found   = sorted({c["meta"].get("city", "") for c in chunks if c["meta"].get("city")})
        print(f"  [RAG] {len(chunks)} fragmente recuperate din: {', '.join(cities_found)}")

    def _build_messages(self) -> list[dict]:
        """
        Assemble the full message list to send to the LLM.

        Structure when RAG is available:
          [system: SYSTEM_PROMPT]
          [user/assistant history ... except last user turn]
          [system: [CONTEXT RAG] block]
          [user: latest user turn]

        This mirrors the training data format exactly.
        """
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.rag_done and self.rag_block:
            # Inject RAG context before the most recent user message
            messages.extend(self.history[:-1])
            messages.append({"role": "system", "content": self.rag_block})
            messages.append(self.history[-1])
        else:
            messages.extend(self.history)

        return messages

    # ------------------------------------------------------------------ #
    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        if self._should_trigger_rag():
            self._do_rag()

        messages = self._build_messages()
        reply    = llm_chat(self.client, messages)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.history   = []
        self.rag_block = None
        self.rag_done  = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialize resources (ChromaDB, embeddings, LLM)
    llm_client, collection, city_map = _init_resources()
    # Start conversation (CLI mode)
    advisor = CityAdvisor(llm_client, collection, city_map)

    print()
    print("=" * 65)
    print("  Asistent de relocare în orașe din România")
    print("  Comenzi speciale: 'reset' — conversație nouă | 'exit' — ieșire")
    print("=" * 65)
    print()

    # Opening greeting — use a direct LLM call so no fake user turn enters history
    opening = llm_chat(
        llm_client,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": "Bună ziua!"}],
    )
    print(f"Asistent: {opening}\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "iesi", "ieși"):
            print("La revedere!")
            break

        if user_input.lower() == "reset":
            advisor.reset()
            print("\n[Conversație resetată]\n")
            opening = llm_chat(
                llm_client,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": "Bună ziua!"}],
            )
            print(f"Asistent: {opening}\n")
            continue

        try:
            reply = advisor.chat(user_input)
            print(f"\nAsistent: {reply}\n")
        except requests.exceptions.ConnectionError:
            print("EROARE: LM Studio nu mai răspunde. Verificați serverul.\n")
        except Exception as exc:
            print(f"EROARE: {exc}\n")


if __name__ == "__main__":
    # Support a web mode: `python chat.py --web` to run a simple Flask API
    if "--web" in sys.argv:
        # Create resources and run a small Flask app exposing the same behavior
        def run_web(host: str = "127.0.0.1", port: int = 5000):
            llm_client, collection, city_map = _init_resources()

            app = Flask(__name__, static_folder=str(REPO_ROOT / "web_console"), static_url_path="")

            sessions: dict[str, CityAdvisor] = {}
            lock = Lock()

            @app.route("/", methods=["GET"])
            def index():
                return app.send_static_file("index.html")

            @app.route("/api/chat", methods=["POST"])
            def api_chat():
                data = request.get_json(force=True) or {}
                prompt = (data.get("prompt") or "").strip()
                if not prompt:
                    return jsonify({"error": "missing prompt"}), 400
                session_id = data.get("session_id")

                if not session_id:
                    session_id = str(uuid.uuid4())
                with lock:
                    if session_id not in sessions:
                        sessions[session_id] = CityAdvisor(llm_client, collection, city_map)
                    advisor = sessions[session_id]

                try:
                    reply = advisor.chat(prompt)
                except Exception as exc:
                    tb = traceback.format_exc()
                    print(f"[ERROR] exception in /api/chat:\n{tb}", file=sys.stderr)
                    # Return limited traceback lines for local debugging
                    tb_lines = tb.splitlines()[-10:]
                    return jsonify({"error": str(exc), "trace": tb_lines}), 500

                return jsonify({"session_id": session_id, "reply": reply})

            @app.route("/api/ping", methods=["GET"])
            def api_ping():
                return jsonify({"ok": True})

            @app.route("/api/reset", methods=["POST"])
            def api_reset():
                data = request.get_json(force=True)
                session_id = data.get("session_id")
                if not session_id:
                    return jsonify({"error": "missing session_id"}), 400
                with lock:
                    if session_id in sessions:
                        sessions[session_id].reset()
                return jsonify({"ok": True})

            app.run(host=host, port=port)

        run_web()
    else:
        main()
