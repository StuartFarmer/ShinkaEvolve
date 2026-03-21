from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Island:
    """
    Computed island domain object.

    Islands are not persisted as first-class rows today. They are reconstructed
    from the `programs` table by grouping on `island_idx`.
    """

    island_idx: int
    total_programs: int = 0
    correct_programs: int = 0
    best_program_id: Optional[str] = None
    best_score: float = 0.0

    @property
    def initialized(self) -> bool:
        return self.correct_programs > 0


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
        num_islands: int,
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.num_islands = num_islands

    def get_program_island(self, program_id: str) -> Optional[int]:
        self.cursor.execute(
            "SELECT island_idx FROM programs WHERE id = ?",
            (program_id,),
        )
        row = self.cursor.fetchone()
        return row["island_idx"] if row else None

    def list_islands(self) -> List[Island]:
        """
        Return computed islands from persisted programs.

        The configured base islands are always represented, even if empty.
        Dynamically spawned islands are included up to the max observed
        `island_idx`.
        """
        max_idx = self.get_max_island_index()
        upper_bound = max(max_idx, self.num_islands - 1)
        if upper_bound < 0:
            return []

        self.cursor.execute(
            """
            SELECT
                island_idx,
                COUNT(*) AS total_programs,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_programs,
                MAX(CASE WHEN correct = 1 THEN combined_score ELSE NULL END) AS best_score
            FROM programs
            WHERE island_idx IS NOT NULL
            GROUP BY island_idx
            """
        )
        rows_by_island = {int(row["island_idx"]): row for row in self.cursor.fetchall()}

        islands: List[Island] = []
        for island_idx in range(upper_bound + 1):
            row = rows_by_island.get(island_idx)
            if row is None:
                islands.append(Island(island_idx=island_idx))
                continue
            best_score_raw = row["best_score"]
            best_program_id = None
            if best_score_raw is not None:
                self.cursor.execute(
                    """
                    SELECT id
                    FROM programs
                    WHERE island_idx = ? AND correct = 1
                    ORDER BY combined_score DESC, timestamp ASC, id ASC
                    LIMIT 1
                    """,
                    (island_idx,),
                )
                best_row = self.cursor.fetchone()
                best_program_id = str(best_row["id"]) if best_row else None

            islands.append(
                Island(
                    island_idx=island_idx,
                    total_programs=int(row["total_programs"] or 0),
                    correct_programs=int(row["correct_programs"] or 0),
                    best_program_id=best_program_id,
                    best_score=float(best_score_raw) if best_score_raw is not None else 0.0,
                )
            )
        return islands

    def get_island(self, island_idx: int) -> Island:
        for island in self.list_islands():
            if island.island_idx == island_idx:
                return island
        return Island(island_idx=island_idx)

    def list_initialized_islands(self) -> List[Island]:
        return [island for island in self.list_islands() if island.initialized]

    def list_initialized_island_ids(self) -> List[int]:
        return [island.island_idx for island in self.list_initialized_islands()]

    def are_all_islands_initialized(self) -> bool:
        if self.num_islands <= 0:
            return True
        return len(self.list_initialized_island_ids()) >= self.num_islands

    def get_island_populations(self) -> Dict[int, int]:
        if self.num_islands <= 0:
            return {}
        return {
            island.island_idx: island.total_programs
            for island in self.list_islands()
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
        return max(max_idx + 1, self.num_islands)

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
