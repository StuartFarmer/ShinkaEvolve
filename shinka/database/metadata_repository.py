from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RunMetadataSnapshot:
    last_iteration: int = 0
    best_program_id: Optional[str] = None
    beam_search_parent_id: Optional[str] = None
    best_score_generation: int = 0
    best_score_ever: Optional[float] = None
    initial_program_count_adjustment: int = 0


class MetadataRepository:
    """
    Persistence boundary for run-level metadata stored alongside programs.

    This repository owns simple key/value metadata and any typed reconstruction
    of that metadata into runtime state objects.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        read_only: bool = False,
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.read_only = read_only

    def ensure_schema(self) -> None:
        if self.read_only:
            return
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_store (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self.conn.commit()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        self.cursor.execute("SELECT value FROM metadata_store WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        if not row:
            return default
        value = row["value"]
        return default if value is None else str(value)

    def set(self, key: str, value: Optional[str]) -> None:
        if self.read_only:
            raise PermissionError("Cannot update metadata in read-only mode.")
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def load_snapshot(self) -> RunMetadataSnapshot:
        last_iteration_raw = self.get("last_iteration")
        if last_iteration_raw is None and not self.read_only:
            self.set("last_iteration", "0")

        best_program_id_raw = self.get("best_program_id")
        if best_program_id_raw in [None, "None"] and not self.read_only:
            self.set("best_program_id", None)

        beam_parent_raw = self.get("beam_search_parent_id")
        if beam_parent_raw in [None, "None"] and not self.read_only:
            self.set("beam_search_parent_id", None)

        best_score_generation_raw = self.get("best_score_generation")
        best_score_ever_raw = self.get("best_score_ever")
        adjustment_raw = self.get("initial_program_count_adjustment")

        if adjustment_raw is None:
            self.cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM programs
                WHERE generation = 0 AND parent_id IS NULL
                """
            )
            row = self.cursor.fetchone()
            initial_root_count = int(row["count"]) if row else 0
            adjustment = max(initial_root_count - 1, 0)
            if not self.read_only:
                self.set("initial_program_count_adjustment", str(adjustment))
        else:
            adjustment = int(adjustment_raw)

        return RunMetadataSnapshot(
            last_iteration=int(last_iteration_raw) if last_iteration_raw is not None else 0,
            best_program_id=None
            if best_program_id_raw in [None, "None"]
            else str(best_program_id_raw),
            beam_search_parent_id=None
            if beam_parent_raw in [None, "None"]
            else str(beam_parent_raw),
            best_score_generation=int(best_score_generation_raw)
            if best_score_generation_raw is not None
            else 0,
            best_score_ever=float(best_score_ever_raw)
            if best_score_ever_raw is not None
            else None,
            initial_program_count_adjustment=adjustment,
        )
