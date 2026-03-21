from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional


class IslandRepository:
    """
    Query boundary for island-scoped persisted facts.

    This repository exposes only island-related storage queries. Island search
    behavior still lives in strategies/services above it.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        config: Any,
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.config = config

    def get_program_island(self, program_id: str) -> Optional[int]:
        self.cursor.execute(
            "SELECT island_idx FROM programs WHERE id = ?",
            (program_id,),
        )
        row = self.cursor.fetchone()
        return row["island_idx"] if row else None

    def list_initialized_islands(self) -> List[int]:
        self.cursor.execute(
            """
            SELECT DISTINCT island_idx
            FROM programs
            WHERE correct = 1 AND island_idx IS NOT NULL
            ORDER BY island_idx ASC
            """
        )
        return [int(row["island_idx"]) for row in self.cursor.fetchall()]

    def are_all_islands_initialized(self) -> bool:
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands <= 0:
            return True
        return len(self.list_initialized_islands()) >= num_islands

    def get_island_populations(self) -> Dict[int, int]:
        if not hasattr(self.config, "num_islands") or self.config.num_islands <= 0:
            return {}
        self.cursor.execute(
            """
            SELECT island_idx, COUNT(id) as count
            FROM programs
            GROUP BY island_idx
            """
        )
        return {
            row["island_idx"]: row["count"]
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }

    def get_program_count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) as count FROM programs")
        row = self.cursor.fetchone()
        return int(row["count"]) if row else 0

    def get_max_island_index(self) -> int:
        self.cursor.execute("SELECT MAX(island_idx) as max_idx FROM programs")
        row = self.cursor.fetchone()
        return row["max_idx"] if row and row["max_idx"] is not None else -1

    def get_next_island_index(self) -> int:
        max_idx = self.get_max_island_index()
        num_islands = getattr(self.config, "num_islands", 1)
        return max(max_idx + 1, num_islands)

    def get_initial_program_row(self) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT * FROM programs
            WHERE generation = 0 AND parent_id IS NULL
            ORDER BY timestamp ASC
            LIMIT 1
            """
        )
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_best_program_row(self) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT * FROM programs
            WHERE correct = 1
            ORDER BY combined_score DESC
            LIMIT 1
            """
        )
        row = self.cursor.fetchone()
        return dict(row) if row else None
