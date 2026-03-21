"""
SQLite-backed repository for persisted `Program` records.

This repository is the storage boundary for the evolution system. It owns:

- database connection / schema setup
- row <-> `Program` serialization
- persistence-oriented queries

It does not own search policy:

- no parent sampling
- no inspiration sampling
- no island sampling
- no novelty logic

Those concerns should depend on the repository, not be embedded inside it.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .complexity import analyze_code_metrics
from .dbase import DatabaseConfig, Program
from .metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


def _clean_nan_values(obj: Any) -> Any:
    """Recursively replace NaN/Inf values so JSON serialization remains valid."""
    if isinstance(obj, dict):
        return {key: _clean_nan_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_values(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_nan_values(item) for item in obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return _clean_nan_values(obj.tolist())
    return obj


@dataclass(frozen=True)
class ProgramCountSnapshot:
    """Lightweight repository snapshot for polling/refresh checks."""

    count: int
    max_timestamp: float | None


class ProgramRepository:
    """
    Persistence-only repository for `Program`.

    The repository uses direct SQLite access and deliberately avoids depending on
    `ProgramDatabase` for core reads/writes. A temporary legacy escape hatch can
    still materialize a `ProgramDatabase` for code paths that have not been
    migrated yet.
    """

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        read_only: bool = False,
    ):
        self.config = config
        self.read_only = read_only
        self.conn: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

        self.last_iteration: int = 0
        self.best_program_id: str | None = None
        self.metadata_repo: MetadataRepository | None = None

        self._connect()
        self._ensure_schema()
        self._load_metadata()

    @classmethod
    def from_config(
        cls,
        config: DatabaseConfig,
        *,
        embedding_model: str = "text-embedding-3-small",
        read_only: bool = False,
    ) -> "ProgramRepository":
        # `embedding_model` is intentionally ignored here. It is kept only so
        # existing call sites can migrate without churn.
        _ = embedding_model
        return cls(config=config, read_only=read_only)

    def _connect(self) -> None:
        db_path_str = getattr(self.config, "db_path", None)

        if db_path_str:
            db_file = Path(db_path_str).resolve()
            if not self.read_only:
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(db_file), timeout=30.0)
            else:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"file:{db_file}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
        else:
            if self.read_only:
                raise ValueError("Read-only repository requires config.db_path")
            self.conn = sqlite3.connect(":memory:", timeout=30.0)

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.metadata_repo = MetadataRepository(
            conn=self.conn,
            cursor=self.cursor,
            read_only=self.read_only,
        )

    def _ensure_schema(self) -> None:
        if not self.cursor or not self.conn:
            raise ConnectionError("Repository not connected.")

        if self.read_only:
            return

        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 30000;")
        self.cursor.execute("PRAGMA wal_autocheckpoint = 1000;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        self.cursor.execute("PRAGMA cache_size = -64000;")
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                archive_inspiration_ids TEXT,
                top_k_inspiration_ids TEXT,
                generation INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                code_diff TEXT,
                combined_score REAL,
                public_metrics TEXT,
                private_metrics TEXT,
                text_feedback TEXT,
                complexity REAL,
                embedding TEXT,
                embedding_pca_2d TEXT,
                embedding_pca_3d TEXT,
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT 0,
                children_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,
                migration_history TEXT,
                island_idx INTEGER,
                system_prompt_id TEXT
            )
            """
        )

        idx_cmds = [
            "CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation)",
            "CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_programs_complexity ON programs(complexity)",
            "CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_programs_children_count ON programs(children_count)",
            "CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON programs(island_idx)",
            "CREATE INDEX IF NOT EXISTS idx_programs_system_prompt_id ON programs(system_prompt_id)",
        ]
        for cmd in idx_cmds:
            self.cursor.execute(cmd)

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archive (
                program_id TEXT PRIMARY KEY,
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        self.metadata_repo.ensure_schema()

        self.conn.commit()
        self._run_migrations()

    def _run_migrations(self) -> None:
        if not self.cursor or not self.conn or self.read_only:
            return

        self.cursor.execute("PRAGMA table_info(programs)")
        columns = [row[1] for row in self.cursor.fetchall()]

        if "text_feedback" not in columns:
            self.cursor.execute(
                "ALTER TABLE programs ADD COLUMN text_feedback TEXT DEFAULT ''"
            )
        if "system_prompt_id" not in columns:
            self.cursor.execute("ALTER TABLE programs ADD COLUMN system_prompt_id TEXT")
        self.conn.commit()

    def _load_metadata(self) -> None:
        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        snapshot = self.metadata_repo.load_snapshot()
        self.last_iteration = snapshot.last_iteration
        self.best_program_id = snapshot.best_program_id

    def _update_metadata(self, key: str, value: Optional[str]) -> None:
        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        self.metadata_repo.set(key, value)

    def get_metadata(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch one metadata value from the repository-local metadata store."""
        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        return self.metadata_repo.get(key, default)

    def set_metadata(self, key: str, value: Optional[str]) -> None:
        """Persist one metadata value."""
        self._update_metadata(key, value)

    def _serialize_json(self, value: Any) -> str:
        return json.dumps(_clean_nan_values(value))

    def _row_to_program(self, row: sqlite3.Row | None) -> Optional[Program]:
        if not row:
            return None

        data = dict(row)

        json_dict_fields = ["public_metrics", "private_metrics", "metadata"]
        json_list_fields = [
            "archive_inspiration_ids",
            "top_k_inspiration_ids",
            "embedding",
            "embedding_pca_2d",
            "embedding_pca_3d",
            "migration_history",
        ]

        for field in json_dict_fields:
            raw = data.get(field)
            if raw:
                try:
                    data[field] = json.loads(raw)
                except json.JSONDecodeError:
                    data[field] = {}
            else:
                data[field] = {}

        for field in json_list_fields:
            raw = data.get(field)
            if raw:
                try:
                    data[field] = json.loads(raw)
                except json.JSONDecodeError:
                    data[field] = []
            else:
                data[field] = []

        if "text_feedback" not in data or data["text_feedback"] is None:
            data["text_feedback"] = ""

        in_archive = data.get("in_archive")
        if in_archive is not None:
            data["in_archive"] = bool(in_archive)

        return Program.from_dict(data)

    def add(self, program: Program, *, verbose: bool = False) -> str:
        """Persist a fully constructed program."""
        if self.read_only:
            raise PermissionError("Cannot add program in read-only mode.")
        if not self.cursor or not self.conn:
            raise ConnectionError("Repository not connected.")

        if program.complexity == 0.0:
            try:
                code_metrics = analyze_code_metrics(program.code, program.language)
                program.complexity = code_metrics.get("complexity_score", 0.0)
                if program.metadata is None:
                    program.metadata = {}
                program.metadata["code_analysis_metrics"] = code_metrics
            except Exception as e:
                logger.warning(
                    "Could not calculate complexity for program %s: %s",
                    program.id,
                    e,
                )
                program.complexity = float(len(program.code))

        if not isinstance(program.embedding, list):
            program.embedding = []

        text_feedback = program.text_feedback
        if isinstance(text_feedback, list):
            text_feedback = "\n".join(str(item) for item in text_feedback)
        elif text_feedback is None:
            text_feedback = ""
        else:
            text_feedback = str(text_feedback)

        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.cursor.execute(
                """
                INSERT INTO programs
                   (id, code, language, parent_id, archive_inspiration_ids,
                    top_k_inspiration_ids, generation, timestamp, code_diff,
                    combined_score, public_metrics, private_metrics,
                    text_feedback, complexity, embedding, embedding_pca_2d,
                    embedding_pca_3d, embedding_cluster_id, correct,
                    children_count, metadata, island_idx, migration_history,
                    system_prompt_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    program.id,
                    program.code,
                    program.language,
                    program.parent_id,
                    self._serialize_json(program.archive_inspiration_ids or []),
                    self._serialize_json(program.top_k_inspiration_ids or []),
                    program.generation,
                    program.timestamp,
                    program.code_diff,
                    program.combined_score,
                    self._serialize_json(program.public_metrics or {}),
                    self._serialize_json(program.private_metrics or {}),
                    text_feedback,
                    program.complexity,
                    self._serialize_json(program.embedding or []),
                    self._serialize_json(program.embedding_pca_2d or []),
                    self._serialize_json(program.embedding_pca_3d or []),
                    program.embedding_cluster_id,
                    program.correct,
                    program.children_count,
                    self._serialize_json(program.metadata or {}),
                    program.island_idx,
                    self._serialize_json(program.migration_history or []),
                    program.system_prompt_id,
                ),
            )

            if program.parent_id:
                self.cursor.execute(
                    "UPDATE programs SET children_count = children_count + 1 WHERE id = ?",
                    (program.parent_id,),
                )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        if program.generation > self.last_iteration:
            self.last_iteration = program.generation
            self._update_metadata("last_iteration", str(self.last_iteration))

        current_best = self.get_best()
        if current_best and current_best.id != self.best_program_id:
            self.best_program_id = current_best.id
            self._update_metadata("best_program_id", self.best_program_id)

        if verbose:
            logger.info(
                "Program %s added to repository - score: %s.",
                program.id,
                program.combined_score,
            )

        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        """Fetch one program by id."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        self.cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
        return self._row_to_program(self.cursor.fetchone())

    def get_children_count(self, program_id: str) -> int:
        """Return the number of persisted children for one program."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        self.cursor.execute(
            "SELECT children_count FROM programs WHERE id = ?",
            (program_id,),
        )
        row = self.cursor.fetchone()
        return int(row["children_count"]) if row else 0

    def get_many(self, program_ids: List[str]) -> List[Program]:
        """Fetch multiple programs, preserving order and skipping missing ids."""
        programs: List[Program] = []
        for program_id in program_ids:
            program = self.get(program_id)
            if program is not None:
                programs.append(program)
        return programs

    def _list_programs(
        self,
        *,
        where_sql: str = "",
        params: Sequence[Any] = (),
        order_sql: str = "ORDER BY generation ASC, timestamp ASC, id ASC",
    ) -> List[Program]:
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        query = f"SELECT * FROM programs {where_sql} {order_sql}".strip()
        self.cursor.execute(query, tuple(params))
        rows = self.cursor.fetchall()
        return [p for p in (self._row_to_program(row) for row in rows) if p is not None]

    def get_best(
        self,
        metric: Optional[str] = None,
        *,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        """Fetch the current best correct program."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")

        programs = self.list_correct(island_idx=island_idx)
        if not programs:
            return None

        if metric:
            eligible = [p for p in programs if p.public_metrics and metric in p.public_metrics]
            sorted_programs = sorted(
                eligible,
                key=lambda p: p.public_metrics.get(metric, -float("inf")),
                reverse=True,
            )
        elif any(p.combined_score is not None for p in programs):
            eligible = [p for p in programs if p.combined_score is not None]
            sorted_programs = sorted(
                eligible,
                key=lambda p: p.combined_score or -float("inf"),
                reverse=True,
            )
        else:
            eligible = [p for p in programs if p.public_metrics]
            sorted_programs = sorted(
                eligible,
                key=lambda p: (
                    sum(p.public_metrics.values()) / len(p.public_metrics)
                    if p.public_metrics
                    else -float("inf")
                ),
                reverse=True,
            )

        if not sorted_programs:
            return None

        best = sorted_programs[0]
        if metric is None and island_idx is None and best.id != self.best_program_id:
            self.best_program_id = best.id
            if not self.read_only:
                self._update_metadata("best_program_id", self.best_program_id)
        return best

    def get_ancestry(self, program_id: str, *, max_ancestors: int = 10) -> List[Program]:
        """Return ancestor programs ordered oldest-to-newest."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")

        ancestors: List[Program] = []
        current_id = program_id
        for _ in range(max_ancestors):
            self.cursor.execute(
                "SELECT parent_id FROM programs WHERE id = ?",
                (current_id,),
            )
            row = self.cursor.fetchone()
            if not row or not row["parent_id"]:
                break
            parent = self.get(row["parent_id"])
            if parent is None:
                break
            ancestors.append(parent)
            current_id = row["parent_id"]
        ancestors.reverse()
        return ancestors

    def list_all(self) -> List[Program]:
        """Return all persisted programs."""
        return self._list_programs()

    def list_correct(self, *, island_idx: Optional[int] = None) -> List[Program]:
        """Return all correct programs, optionally constrained to one island."""
        where = "WHERE correct = 1"
        params: List[Any] = []
        if island_idx is not None:
            where += " AND island_idx = ?"
            params.append(island_idx)
        return self._list_programs(where_sql=where, params=params)

    def list_incorrect(self, *, island_idx: Optional[int] = None) -> List[Program]:
        """Return all incorrect programs, optionally constrained to one island."""
        where = "WHERE correct = 0"
        params: List[Any] = []
        if island_idx is not None:
            where += " AND island_idx = ?"
            params.append(island_idx)
        return self._list_programs(where_sql=where, params=params)

    def list_by_island(
        self,
        island_idx: int,
        *,
        correct_only: bool = False,
    ) -> List[Program]:
        """Return programs assigned to one island."""
        where = "WHERE island_idx = ?"
        params: List[Any] = [island_idx]
        if correct_only:
            where += " AND correct = 1"
        return self._list_programs(where_sql=where, params=params)

    def list_initialized_islands(self) -> List[int]:
        """Return islands that currently have at least one correct program."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        self.cursor.execute(
            """
            SELECT DISTINCT island_idx
            FROM programs
            WHERE correct = 1 AND island_idx IS NOT NULL
            ORDER BY island_idx ASC
            """
        )
        return [int(row["island_idx"]) for row in self.cursor.fetchall()]

    def get_island_program_counts(
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, int]:
        """Return correct-program counts for a set of islands."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        if not island_indices:
            return {}
        placeholders = ",".join("?" * len(island_indices))
        self.cursor.execute(
            f"""
            SELECT island_idx, COUNT(*) as count
            FROM programs
            WHERE island_idx IN ({placeholders}) AND correct = 1
            GROUP BY island_idx
            """,
            tuple(island_indices),
        )
        counts = {int(island_idx): 0 for island_idx in island_indices}
        for row in self.cursor.fetchall():
            counts[int(row["island_idx"])] = int(row["count"])
        return counts

    def get_island_best_scores(
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, float]:
        """Return best combined score per island for a set of islands."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        if not island_indices:
            return {}
        placeholders = ",".join("?" * len(island_indices))
        self.cursor.execute(
            f"""
            SELECT island_idx, MAX(combined_score) as best_fitness
            FROM programs
            WHERE island_idx IN ({placeholders}) AND correct = 1
            GROUP BY island_idx
            """,
            tuple(island_indices),
        )
        scores: Dict[int, float] = {}
        for row in self.cursor.fetchall():
            value = row["best_fitness"]
            scores[int(row["island_idx"])] = float(value) if value is not None else 0.0
        return scores

    def get_earliest(
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        """Return the earliest persisted program matching the requested scope."""
        where_clauses: List[str] = []
        params: List[Any] = []
        if correct_only:
            where_clauses.append("correct = 1")
        if island_idx is not None:
            where_clauses.append("island_idx = ?")
            params.append(island_idx)
        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        programs = self._list_programs(
            where_sql=where,
            params=params,
            order_sql="ORDER BY generation ASC, timestamp ASC, id ASC LIMIT 1",
        )
        return programs[0] if programs else None

    def get_most_recent(
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        """Return the most recent persisted program matching the requested scope."""
        where_clauses: List[str] = []
        params: List[Any] = []
        if correct_only:
            where_clauses.append("correct = 1")
        if island_idx is not None:
            where_clauses.append("island_idx = ?")
            params.append(island_idx)
        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        programs = self._list_programs(
            where_sql=where,
            params=params,
            order_sql="ORDER BY generation DESC, timestamp DESC, id DESC LIMIT 1",
        )
        return programs[0] if programs else None

    def list_by_generation(self, generation: int) -> List[Program]:
        """Return all programs from one generation."""
        return self._list_programs(where_sql="WHERE generation = ?", params=[generation])

    def list_top(
        self,
        *,
        n: int = 10,
        metric: Optional[str] = "combined_score",
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> List[Program]:
        """Return top programs ordered by the requested metric."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")

        clauses: List[str] = []
        params: List[Any] = []
        if correct_only:
            clauses.append("correct = 1")
        if island_idx is not None:
            clauses.append("island_idx = ?")
            params.append(island_idx)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        if metric == "combined_score":
            query = "SELECT * FROM programs WHERE combined_score IS NOT NULL"
            if clauses:
                query += " AND " + " AND ".join(clauses)
            query += " ORDER BY combined_score DESC LIMIT ?"
            self.cursor.execute(query, (*params, n))
            rows = self.cursor.fetchall()
            return [p for p in (self._row_to_program(row) for row in rows) if p is not None]

        if metric == "timestamp":
            query = f"SELECT * FROM programs {where_sql} ORDER BY timestamp DESC LIMIT ?"
            self.cursor.execute(query, (*params, n))
            rows = self.cursor.fetchall()
            return [p for p in (self._row_to_program(row) for row in rows) if p is not None]

        query = f"SELECT * FROM programs {where_sql}"
        self.cursor.execute(query, tuple(params))
        rows = self.cursor.fetchall()
        programs = [p for p in (self._row_to_program(row) for row in rows) if p is not None]
        if not programs:
            return []

        if metric:
            eligible = [p for p in programs if p.public_metrics and metric in p.public_metrics]
            sorted_programs = sorted(
                eligible,
                key=lambda p: p.public_metrics.get(metric, -float("inf")),
                reverse=True,
            )
        else:
            eligible = [p for p in programs if p.public_metrics]
            sorted_programs = sorted(
                eligible,
                key=lambda p: (
                    sum(p.public_metrics.values()) / len(p.public_metrics)
                    if p.public_metrics
                    else -float("inf")
                ),
                reverse=True,
            )
        return sorted_programs[:n]

    def get_summaries(self) -> List[dict[str, Any]]:
        """Return lightweight summaries for visualization and polling."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")

        self.cursor.execute(
            """
            SELECT
                p.id,
                p.parent_id,
                p.generation,
                p.timestamp,
                p.combined_score,
                p.correct,
                p.complexity,
                p.island_idx,
                p.children_count,
                p.public_metrics,
                p.private_metrics,
                p.metadata,
                p.embedding_pca_2d,
                p.embedding_pca_3d,
                p.embedding_cluster_id,
                p.language,
                p.top_k_inspiration_ids,
                p.archive_inspiration_ids,
                p.migration_history
            FROM programs p
            """
        )
        rows = self.cursor.fetchall()
        summaries: List[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in ["public_metrics", "private_metrics", "metadata"]:
                raw = item.get(field)
                if raw:
                    try:
                        item[field] = json.loads(raw)
                    except json.JSONDecodeError:
                        item[field] = {}
                else:
                    item[field] = {}
            for field in [
                "embedding_pca_2d",
                "embedding_pca_3d",
                "top_k_inspiration_ids",
                "archive_inspiration_ids",
                "migration_history",
            ]:
                raw = item.get(field)
                if raw:
                    try:
                        item[field] = json.loads(raw)
                    except json.JSONDecodeError:
                        item[field] = []
                else:
                    item[field] = []
            item["in_archive"] = False
            summaries.append(item)
        return summaries

    def get_count_snapshot(self) -> ProgramCountSnapshot:
        """Return a cheap count/timestamp summary of the repository."""
        if not self.cursor:
            raise ConnectionError("Repository not connected.")
        self.cursor.execute(
            "SELECT COUNT(*) as count, MAX(timestamp) as max_timestamp FROM programs"
        )
        row = self.cursor.fetchone()
        return ProgramCountSnapshot(
            count=int(row["count"]) if row else 0,
            max_timestamp=row["max_timestamp"] if row else None,
        )

    @property
    def db(self):
        """
        Temporary legacy escape hatch.

        This intentionally exists only to support incremental migration of code
        paths that still expect `ProgramDatabase`. New code should not use it.
        """
        from .dbase import ProgramDatabase

        return ProgramDatabase(self.config, read_only=self.read_only)

    def close(self) -> None:
        """Close the underlying storage connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
