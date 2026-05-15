"""
progress_tracker.py
Checkpoint system for resumable synthetic data generation.
Every successful API call is recorded immediately so the process is safe
to interrupt at any point and re-run when API limits reset.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILE = Path(__file__).resolve().parents[1] / "data" / "generation_progress.json"
RAW_CONVERSATIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "raw_conversations.jsonl"

SCHEMA_VERSION = "1.0"
MAX_RETRIES = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ProgressTracker:
    """
    Tracks which scenarios have been generated so that re-runs skip completed work.

    Usage
    -----
    tracker = ProgressTracker()
    run_id = tracker.start_run()
    for scenario in pending_scenarios:
        if tracker.is_done(scenario["scenario_id"]):
            continue
        conversation = call_api(scenario)
        tracker.record_success(scenario["scenario_id"], conversation, run_id)
    tracker.finish_run(run_id)
    """

    def __init__(
        self,
        progress_file: Path = PROGRESS_FILE,
        output_file: Path = RAW_CONVERSATIONS_FILE,
    ) -> None:
        self.progress_file = Path(progress_file)
        self.output_file = Path(output_file)
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        state = _load(self.progress_file)
        if not state:
            state = {
                "schema_version": SCHEMA_VERSION,
                "completed_scenario_ids": [],
                "failed_scenario_ids": [],
                "retry_counts": {},
                "run_history": [],
                "output_file": str(self.output_file),
            }
        # Ensure sets are used in-memory for fast lookup
        state["_completed_set"] = set(state.get("completed_scenario_ids", []))
        state["_failed_set"] = set(state.get("failed_scenario_ids", []))
        return state

    def _persist(self) -> None:
        # Sync back from working sets before saving
        self._state["completed_scenario_ids"] = sorted(self._state["_completed_set"])
        self._state["failed_scenario_ids"] = sorted(self._state["_failed_set"])
        save_state = {k: v for k, v in self._state.items() if not k.startswith("_")}
        _save(self.progress_file, save_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self) -> str:
        run_id = f"run_{str(uuid.uuid4())[:8]}"
        self._state["run_history"].append(
            {"run_id": run_id, "started": _now_iso(), "completed": 0, "failed": 0}
        )
        self._persist()
        return run_id

    def finish_run(self, run_id: str) -> None:
        for entry in self._state["run_history"]:
            if entry["run_id"] == run_id:
                entry["finished"] = _now_iso()
        self._persist()

    def is_done(self, scenario_id: str) -> bool:
        return scenario_id in self._state["_completed_set"]

    def is_exhausted(self, scenario_id: str) -> bool:
        """Returns True if the scenario has failed too many times to retry."""
        return (
            scenario_id in self._state["_failed_set"]
            and self._state["retry_counts"].get(scenario_id, 0) >= MAX_RETRIES
        )

    def record_success(
        self,
        scenario_id: str,
        conversation: dict[str, Any],
        run_id: str,
    ) -> None:
        """Appends the conversation to the output JSONL and marks the scenario done."""
        # Write conversation to JSONL (append mode — never overwrites)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        # Update checkpoint
        self._state["_completed_set"].add(scenario_id)
        self._state["_failed_set"].discard(scenario_id)

        # Increment run counter
        for entry in self._state["run_history"]:
            if entry["run_id"] == run_id:
                entry["completed"] = entry.get("completed", 0) + 1

        self._persist()

    def record_failure(self, scenario_id: str, run_id: str, error: str = "") -> None:
        counts = self._state.setdefault("retry_counts", {})
        counts[scenario_id] = counts.get(scenario_id, 0) + 1
        if counts[scenario_id] >= MAX_RETRIES:
            self._state["_failed_set"].add(scenario_id)

        for entry in self._state["run_history"]:
            if entry["run_id"] == run_id:
                entry["failed"] = entry.get("failed", 0) + 1

        self._persist()

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @property
    def completed_count(self) -> int:
        return len(self._state["_completed_set"])

    @property
    def failed_count(self) -> int:
        return len(self._state["_failed_set"])

    def summary(self) -> str:
        return (
            f"Completed: {self.completed_count} | "
            f"Permanently failed: {self.failed_count} | "
            f"Runs: {len(self._state['run_history'])}"
        )
