"""
ORM-backed repository for persisted `Program` records.

The repository is the storage boundary for the evolution system. It owns:

- connection / session wiring
- schema bootstrap for `programs`
- `Program` <-> `ProgramRecord` mapping
- persistence-oriented queries and mutations

It does not own search policy.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import create_engine, func, select, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .complexity import analyze_code_metrics
from .program import Program
from .inspiration_repository import InspirationRepository, InspirationUse
from .island_repository import Island, IslandRepository
from .metadata_repository import MetadataRepository
from .models import (
    Base,
    ProgramEmbeddingProjectionRecord,
    ProgramEmbeddingRecord,
    ProgramEvaluationRecord,
    ProgramProposalRecord,
    ProgramRecord,
)

logger = logging.getLogger(__name__)


def _clean_nan_values(obj: Any) -> Any:
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
            return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        return _clean_nan_values(obj.tolist())
    return obj


def _normalize_text_feedback(text_feedback: Any) -> str:
    if isinstance(text_feedback, list):
        return "\n".join(str(item) for item in text_feedback)
    if text_feedback is None:
        return ""
    return str(text_feedback)


@dataclass(frozen=True)
class ProgramCountSnapshot:
    count: int
    max_timestamp: float | None


class ProgramRepository:
    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        num_islands: int = 2,
        read_only: bool = False,
    ):
        self.db_path = db_path
        self.num_islands = num_islands
        self.read_only = read_only
        self.conn: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.engine = None
        self.SessionLocal = None

        self.last_iteration: int = 0
        self.best_program_id: str | None = None
        self.metadata_repo: MetadataRepository | None = None
        self.island_repo: IslandRepository | None = None
        self.inspiration_repo: InspirationRepository | None = None

        self._connect()
        self._ensure_schema()
        self._load_metadata()

    @classmethod
    def from_db_path(
        cls,
        db_path: str,
        *,
        num_islands: int = 0,
        read_only: bool = False,
    ) -> ProgramRepository:
        return cls(db_path=db_path, num_islands=num_islands, read_only=read_only)

    @classmethod
    def from_existing_connection(
        cls,
        *,
        db_path: Optional[str] = None,
        num_islands: int = 2,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        read_only: bool = False,
        ensure_schema: bool = False,
    ) -> ProgramRepository:
        repo = cls.__new__(cls)
        repo.db_path = db_path
        repo.num_islands = num_islands
        repo.read_only = read_only
        repo.conn = conn
        repo.cursor = cursor
        repo.engine = create_engine(
            "sqlite://",
            creator=lambda: conn,
            poolclass=StaticPool,
            future=True,
        )
        repo.SessionLocal = sessionmaker(
            bind=repo.engine,
            expire_on_commit=False,
            future=True,
        )
        repo.last_iteration = 0
        repo.best_program_id = None
        repo.metadata_repo = MetadataRepository(
            conn=conn,
            cursor=cursor,
            read_only=read_only,
        )
        repo.inspiration_repo = InspirationRepository(
            session_factory=repo.SessionLocal,
            read_only=read_only,
        )
        repo.island_repo = IslandRepository(
            conn=conn,
            cursor=cursor,
            num_islands=num_islands,
        )
        if ensure_schema and not read_only:
            repo._ensure_schema()
        repo._load_metadata()
        return repo

    def _connect(self) -> None:
        db_path_str = self.db_path
        if db_path_str:
            db_file = Path(db_path_str).resolve()
            if self.read_only:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                self.conn = sqlite3.connect(
                    f"file:{db_file}?mode=ro",
                    uri=True,
                    timeout=30.0,
                )
            else:
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(db_file), timeout=30.0)
        else:
            if self.read_only:
                raise ValueError("Read-only repository requires db_path")
            self.conn = sqlite3.connect(":memory:", timeout=30.0)

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.engine = create_engine(
            "sqlite://",
            creator=lambda: self.conn,
            poolclass=StaticPool,
            future=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )
        self.metadata_repo = MetadataRepository(
            conn=self.conn,
            cursor=self.cursor,
            read_only=self.read_only,
        )
        self.inspiration_repo = InspirationRepository(
            session_factory=self.SessionLocal,
            read_only=self.read_only,
        )
        self.island_repo = IslandRepository(
            conn=self.conn,
            cursor=self.cursor,
            num_islands=self.num_islands,
        )

    def _ensure_schema(self) -> None:
        if self.read_only:
            return
        if not self.conn or not self.cursor or self.engine is None:
            raise ConnectionError("Repository not connected.")

        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 30000;")
        self.cursor.execute("PRAGMA wal_autocheckpoint = 1000;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        self.cursor.execute("PRAGMA cache_size = -64000;")
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")

        Base.metadata.create_all(self.engine)
        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        self.metadata_repo.ensure_schema()
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
        if self.metadata_repo is None:
            raise ConnectionError("Repository metadata store not initialized.")
        return self.metadata_repo.get(key, default)

    def set_metadata(self, key: str, value: Optional[str]) -> None:
        self._update_metadata(key, value)

    def _session(self) -> Session:
        if self.SessionLocal is None:
            raise ConnectionError("Repository not connected.")
        return self.SessionLocal()

    def _build_inspiration_index(
        self,
        child_program_ids: Sequence[str],
    ) -> Dict[str, Dict[str, List[str]]]:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        uses_by_child = self.inspiration_repo.list_for_children(child_program_ids)
        index: Dict[str, Dict[str, List[str]]] = {
            child_id: {"archive": [], "top_k": [], "ancestor": []}
            for child_id in child_program_ids
        }
        for child_id, inspirations in uses_by_child.items():
            for inspiration in inspirations:
                role_bucket = index.setdefault(
                    child_id,
                    {"archive": [], "top_k": [], "ancestor": []},
                )
                role_bucket.setdefault(inspiration.role, []).append(
                    inspiration.source_program_id
                )
        return index

    def _program_inspiration_uses(self, program: Program) -> List[InspirationUse]:
        inspirations: List[InspirationUse] = []
        for idx, source_program_id in enumerate(program.archive_inspiration_ids or []):
            inspirations.append(
                InspirationUse(
                    child_program_id=program.id,
                    source_program_id=source_program_id,
                    role="archive",
                    order_index=idx,
                )
            )
        for idx, source_program_id in enumerate(program.top_k_inspiration_ids or []):
            inspirations.append(
                InspirationUse(
                    child_program_id=program.id,
                    source_program_id=source_program_id,
                    role="top_k",
                    order_index=idx,
                )
            )
        return inspirations

    def _build_evaluation_index(
        self,
        program_ids: Sequence[str],
    ) -> Dict[str, ProgramEvaluationRecord]:
        ids = [program_id for program_id in program_ids if program_id]
        if not ids:
            return {}
        with self._session() as session:
            records = session.execute(
                select(ProgramEvaluationRecord).where(
                    ProgramEvaluationRecord.program_id.in_(ids)
                )
            ).scalars().all()
        return {record.program_id: record for record in records}

    def _build_proposal_index(
        self,
        program_ids: Sequence[str],
    ) -> Dict[str, ProgramProposalRecord]:
        ids = [program_id for program_id in program_ids if program_id]
        if not ids:
            return {}
        with self._session() as session:
            records = session.execute(
                select(ProgramProposalRecord).where(
                    ProgramProposalRecord.program_id.in_(ids)
                )
            ).scalars().all()
        return {record.program_id: record for record in records}

    def _build_embedding_index(
        self,
        program_ids: Sequence[str],
    ) -> Dict[str, ProgramEmbeddingRecord]:
        ids = [program_id for program_id in program_ids if program_id]
        if not ids:
            return {}
        with self._session() as session:
            records = session.execute(
                select(ProgramEmbeddingRecord).where(
                    ProgramEmbeddingRecord.program_id.in_(ids)
                )
            ).scalars().all()
        return {record.program_id: record for record in records}

    def _build_projection_index(
        self,
        program_ids: Sequence[str],
    ) -> Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]:
        ids = [program_id for program_id in program_ids if program_id]
        if not ids:
            return {}
        with self._session() as session:
            records = session.execute(
                select(ProgramEmbeddingProjectionRecord).where(
                    ProgramEmbeddingProjectionRecord.program_id.in_(ids)
                )
            ).scalars().all()
        index: Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]] = {
            program_id: {} for program_id in ids
        }
        for record in records:
            index.setdefault(record.program_id, {})[record.kind] = record
        return index

    def _record_to_program(
        self,
        record: ProgramRecord | None,
        *,
        inspiration_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
        evaluation_index: Optional[Dict[str, ProgramEvaluationRecord]] = None,
        proposal_index: Optional[Dict[str, ProgramProposalRecord]] = None,
        embedding_index: Optional[Dict[str, ProgramEmbeddingRecord]] = None,
        projection_index: Optional[Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]] = None,
    ) -> Optional[Program]:
        if record is None:
            return None
        if evaluation_index is None:
            evaluation_index = self._build_evaluation_index([record.id])
        if proposal_index is None:
            proposal_index = self._build_proposal_index([record.id])
        if embedding_index is None:
            embedding_index = self._build_embedding_index([record.id])
        if projection_index is None:
            projection_index = self._build_projection_index([record.id])
        inspiration_lists = (
            inspiration_index.get(record.id, {})
            if inspiration_index is not None
            else self._build_inspiration_index([record.id]).get(record.id, {})
        )
        evaluation = evaluation_index.get(record.id)
        proposal = proposal_index.get(record.id)
        embedding = embedding_index.get(record.id)
        projections = projection_index.get(record.id, {})
        projection_2d = projections.get("pca_2d")
        projection_3d = projections.get("pca_3d")
        diagnostics = (
            dict(evaluation.diagnostics_json or {})
            if evaluation is not None
            else {}
        )
        metadata = dict(record.program_metadata or {})
        if diagnostics.get("code_analysis_metrics") and "code_analysis_metrics" not in metadata:
            metadata["code_analysis_metrics"] = diagnostics["code_analysis_metrics"]
        return Program.from_dict(
            {
                "id": record.id,
                "code": record.code,
                "language": record.language,
                "parent_id": record.parent_id,
                "archive_inspiration_ids": inspiration_lists.get("archive", []),
                "top_k_inspiration_ids": inspiration_lists.get("top_k", []),
                "generation": record.generation,
                "timestamp": record.timestamp,
                "code_diff": proposal.code_diff if proposal is not None else None,
                "combined_score": evaluation.combined_score if evaluation is not None else 0.0,
                "public_metrics": evaluation.public_metrics_json if evaluation is not None else {},
                "private_metrics": evaluation.private_metrics_json if evaluation is not None else {},
                "text_feedback": evaluation.text_feedback if evaluation is not None else "",
                "complexity": evaluation.complexity if evaluation is not None else 0.0,
                "embedding": embedding.vector_json if embedding is not None else [],
                "embedding_pca_2d": projection_2d.coords_json if projection_2d is not None else [],
                "embedding_pca_3d": projection_3d.coords_json if projection_3d is not None else [],
                "embedding_cluster_id": (
                    projection_3d.cluster_id
                    if projection_3d is not None and projection_3d.cluster_id is not None
                    else (projection_2d.cluster_id if projection_2d is not None else None)
                ),
                "correct": bool(evaluation.correct) if evaluation is not None else False,
                "children_count": record.children_count,
                "metadata": metadata,
                "island_idx": record.island_idx,
                "migration_history": record.migration_history or [],
                "system_prompt_id": record.system_prompt_id,
            }
        )

    def _record_to_summary(
        self,
        record: ProgramRecord,
        *,
        inspiration_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
        evaluation_index: Optional[Dict[str, ProgramEvaluationRecord]] = None,
        proposal_index: Optional[Dict[str, ProgramProposalRecord]] = None,
        projection_index: Optional[Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]] = None,
    ) -> dict[str, Any]:
        if evaluation_index is None:
            evaluation_index = self._build_evaluation_index([record.id])
        if proposal_index is None:
            proposal_index = self._build_proposal_index([record.id])
        if projection_index is None:
            projection_index = self._build_projection_index([record.id])
        inspiration_lists = (
            inspiration_index.get(record.id, {})
            if inspiration_index is not None
            else self._build_inspiration_index([record.id]).get(record.id, {})
        )
        evaluation = evaluation_index.get(record.id)
        proposal = proposal_index.get(record.id)
        projections = projection_index.get(record.id, {})
        projection_2d = projections.get("pca_2d")
        projection_3d = projections.get("pca_3d")
        diagnostics = (
            dict(evaluation.diagnostics_json or {})
            if evaluation is not None
            else {}
        )
        metadata = dict(record.program_metadata or {})
        if diagnostics.get("code_analysis_metrics") and "code_analysis_metrics" not in metadata:
            metadata["code_analysis_metrics"] = diagnostics["code_analysis_metrics"]
        return {
            "id": record.id,
            "parent_id": record.parent_id,
            "generation": record.generation,
            "timestamp": record.timestamp,
            "combined_score": evaluation.combined_score if evaluation is not None else None,
            "correct": bool(evaluation.correct) if evaluation is not None else False,
            "complexity": evaluation.complexity if evaluation is not None else 0.0,
            "island_idx": record.island_idx,
            "children_count": record.children_count,
            "public_metrics": evaluation.public_metrics_json if evaluation is not None else {},
            "private_metrics": evaluation.private_metrics_json if evaluation is not None else {},
            "metadata": metadata,
            "embedding_pca_2d": projection_2d.coords_json if projection_2d is not None else [],
            "embedding_pca_3d": projection_3d.coords_json if projection_3d is not None else [],
            "embedding_cluster_id": (
                projection_3d.cluster_id
                if projection_3d is not None and projection_3d.cluster_id is not None
                else (projection_2d.cluster_id if projection_2d is not None else None)
            ),
            "language": record.language,
            "code_diff": proposal.code_diff if proposal is not None else None,
            "top_k_inspiration_ids": inspiration_lists.get("top_k", []),
            "archive_inspiration_ids": inspiration_lists.get("archive", []),
            "migration_history": record.migration_history or [],
            "in_archive": False,
        }

    def _program_query(
        self,
        *,
        correct_only: Optional[bool] = None,
        island_idx: Optional[int] = None,
        generation: Optional[int] = None,
        parent_id: Optional[str] = None,
    ):
        query = select(ProgramRecord).outerjoin(
            ProgramEvaluationRecord,
            ProgramEvaluationRecord.program_id == ProgramRecord.id,
        )
        if correct_only is True:
            query = query.where(ProgramEvaluationRecord.correct.is_(True))
        elif correct_only is False:
            query = query.where(
                (ProgramEvaluationRecord.correct.is_(False))
                | (ProgramEvaluationRecord.correct.is_(None))
            )
        if island_idx is not None:
            query = query.where(ProgramRecord.island_idx == island_idx)
        if generation is not None:
            query = query.where(ProgramRecord.generation == generation)
        if parent_id is not None:
            query = query.where(ProgramRecord.parent_id == parent_id)
        return query

    def _list_programs(self, query) -> List[Program]:
        with self._session() as session:
            records = session.execute(query).scalars().all()
        program_ids = [record.id for record in records]
        inspiration_index = self._build_inspiration_index(program_ids)
        evaluation_index = self._build_evaluation_index(program_ids)
        proposal_index = self._build_proposal_index(program_ids)
        embedding_index = self._build_embedding_index(program_ids)
        projection_index = self._build_projection_index(program_ids)
        return [
            p
            for p in (
                self._record_to_program(
                    record,
                    inspiration_index=inspiration_index,
                    evaluation_index=evaluation_index,
                    proposal_index=proposal_index,
                    embedding_index=embedding_index,
                    projection_index=projection_index,
                )
                for record in records
            )
            if p is not None
        ]

    def add(self, program: Program, *, verbose: bool = False) -> str:
        if self.read_only:
            raise PermissionError("Cannot add program in read-only mode.")

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

        with self._session() as session:
            session.add(
                ProgramRecord(
                    id=program.id,
                    name=getattr(program, "name", None),
                    code=program.code,
                    language=program.language,
                    parent_id=program.parent_id,
                    generation=program.generation,
                    timestamp=program.timestamp,
                    children_count=program.children_count,
                    program_metadata=_clean_nan_values(program.metadata or {}),
                    island_idx=program.island_idx,
                    migration_history=_clean_nan_values(program.migration_history or []),
                    system_prompt_id=program.system_prompt_id,
                )
            )
            session.flush()
            diagnostics = {}
            if program.metadata and program.metadata.get("code_analysis_metrics"):
                diagnostics["code_analysis_metrics"] = _clean_nan_values(
                    program.metadata.get("code_analysis_metrics")
                )
            session.add(
                ProgramEvaluationRecord(
                    id=str(uuid.uuid4()),
                    program_id=program.id,
                    correct=bool(program.correct),
                    combined_score=program.combined_score,
                    text_feedback=_normalize_text_feedback(program.text_feedback),
                    complexity=program.complexity,
                    compute_time_seconds=(
                        float((program.metadata or {}).get("compute_time"))
                        if (program.metadata or {}).get("compute_time") is not None
                        else None
                    ),
                    public_metrics_json=_clean_nan_values(program.public_metrics or {}),
                    private_metrics_json=_clean_nan_values(program.private_metrics or {}),
                    diagnostics_json=_clean_nan_values(diagnostics),
                )
            )
            session.add(
                ProgramProposalRecord(
                    id=str(uuid.uuid4()),
                    program_id=program.id,
                    patch_type=(program.metadata or {}).get("patch_type"),
                    patch_name=(program.metadata or {}).get("patch_name"),
                    code_diff=program.code_diff,
                    system_prompt_id=program.system_prompt_id,
                    proposal_metadata_json=_clean_nan_values(
                        {
                            key: value
                            for key, value in (program.metadata or {}).items()
                            if key
                            not in {
                                "code_analysis_metrics",
                                "compute_time",
                            }
                        }
                    ),
                )
            )
            if program.embedding:
                embedding_id = str(uuid.uuid4())
                session.add(
                    ProgramEmbeddingRecord(
                        id=embedding_id,
                        program_id=program.id,
                        vector_json=_clean_nan_values(program.embedding or []),
                        model_name=(program.metadata or {}).get("embedding_model"),
                        embedding_metadata_json={},
                    )
                )
                if program.embedding_pca_2d:
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=program.id,
                            embedding_id=embedding_id,
                            kind="pca_2d",
                            coords_json=_clean_nan_values(program.embedding_pca_2d or []),
                            cluster_id=program.embedding_cluster_id,
                            projection_metadata_json={},
                        )
                    )
                if program.embedding_pca_3d:
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=program.id,
                            embedding_id=embedding_id,
                            kind="pca_3d",
                            coords_json=_clean_nan_values(program.embedding_pca_3d or []),
                            cluster_id=program.embedding_cluster_id,
                            projection_metadata_json={},
                        )
                    )
            if program.parent_id:
                session.execute(
                    update(ProgramRecord)
                    .where(ProgramRecord.id == program.parent_id)
                    .values(children_count=ProgramRecord.children_count + 1)
                )
            if self.inspiration_repo is None:
                raise ConnectionError("Repository inspiration store not initialized.")
            self.inspiration_repo.replace_for_child(
                program.id,
                self._program_inspiration_uses(program),
                session=session,
            )
            session.commit()

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
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
        return self._record_to_program(record)

    def get_inspiration_uses(
        self,
        child_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> List[InspirationUse]:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        inspirations = self.inspiration_repo.list_for_child(child_program_id)
        if role is not None:
            inspirations = [insp for insp in inspirations if insp.role == role]
        return inspirations

    def get_inspiration_source_ids(
        self,
        child_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> List[str]:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        return self.inspiration_repo.list_sources_for_child(
            child_program_id,
            role=role,
        )

    def get_inspired_child_ids(
        self,
        source_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> List[str]:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        return self.inspiration_repo.list_children_for_source(
            source_program_id,
            role=role,
        )

    def count_inspiration_usage_by_source(
        self,
        source_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> int:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        return self.inspiration_repo.count_usage_by_source(
            source_program_id,
            role=role,
        )

    def count_inspiration_usage_by_role(self, role: str) -> int:
        if self.inspiration_repo is None:
            raise ConnectionError("Repository inspiration store not initialized.")
        return self.inspiration_repo.count_usage_by_role(role)

    def get_children_count(self, program_id: str) -> int:
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
            return int(record.children_count) if record else 0

    def get_many(self, program_ids: List[str]) -> List[Program]:
        if not program_ids:
            return []
        with self._session() as session:
            records = list(
                session.execute(
                select(ProgramRecord).where(ProgramRecord.id.in_(program_ids))
            ).scalars()
            )
        record_ids = [record.id for record in records]
        inspiration_index = self._build_inspiration_index(record_ids)
        evaluation_index = self._build_evaluation_index(record_ids)
        proposal_index = self._build_proposal_index(record_ids)
        embedding_index = self._build_embedding_index(record_ids)
        projection_index = self._build_projection_index(record_ids)
        by_id = {
            record.id: self._record_to_program(
                record,
                inspiration_index=inspiration_index,
                evaluation_index=evaluation_index,
                proposal_index=proposal_index,
                embedding_index=embedding_index,
                projection_index=projection_index,
            )
            for record in records
        }
        return [by_id[program_id] for program_id in program_ids if by_id.get(program_id) is not None]

    def get_program_count(self) -> int:
        if self.island_repo is None:
            raise ConnectionError("Repository island view not initialized.")
        return self.island_repo.get_program_count()

    def count_by_island(self, island_idx: int) -> int:
        with self._session() as session:
            return int(
                session.scalar(
                    select(func.count()).select_from(ProgramRecord).where(
                        ProgramRecord.island_idx == island_idx
                    )
                )
                or 0
            )

    def get_initial_program_row(self) -> Optional[dict[str, Any]]:
        with self._session() as session:
            record = session.scalar(
                select(ProgramRecord)
                .where(
                    ProgramRecord.generation == 0,
                    ProgramRecord.parent_id.is_(None),
                )
                .order_by(ProgramRecord.timestamp.asc())
                .limit(1)
            )
        return None if record is None else self._record_to_program(record).to_dict()

    def get_best_program_row(self) -> Optional[dict[str, Any]]:
        best = self.get_best()
        return None if best is None else best.to_dict()

    def get_next_island_index(self) -> int:
        if self.island_repo is None:
            raise ConnectionError("Repository island view not initialized.")
        return self.island_repo.get_next_island_index()

    def list_migrant_ids(
        self,
        *,
        source_idx: int,
        num_migrants: int,
        island_elitism: bool,
    ) -> List[str]:
        with self._session() as session:
            base = (
                select(ProgramRecord.id)
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
                .where(
                    ProgramRecord.island_idx == source_idx,
                    ProgramRecord.generation > 0,
                    ProgramEvaluationRecord.correct.is_(True),
                )
            )
            available = int(
                session.scalar(
                    select(func.count())
                    .select_from(ProgramRecord)
                    .join(
                        ProgramEvaluationRecord,
                        ProgramEvaluationRecord.program_id == ProgramRecord.id,
                    )
                    .where(
                        ProgramRecord.island_idx == source_idx,
                        ProgramRecord.generation > 0,
                        ProgramEvaluationRecord.correct.is_(True),
                    )
                )
                or 0
            )
            if available == 0:
                return []
            if island_elitism:
                elite_id = session.scalar(
                    base.order_by(ProgramEvaluationRecord.combined_score.desc()).limit(1)
                )
                if elite_id:
                    base = base.where(ProgramRecord.id != elite_id)
            return list(
                dict.fromkeys(
                    session.execute(
                        base.order_by(func.random()).limit(min(num_migrants, available))
                    ).scalars().all()
                )
            )

    def migrate_program(
        self,
        *,
        migrant_id: str,
        source_idx: int,
        dest_idx: int,
        current_generation: int,
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot migrate program in read-only mode.")
        with self._session() as session:
            record = session.get(ProgramRecord, migrant_id)
            if record is None:
                return
            history = list(record.migration_history or [])
            history.append(
                {
                    "generation": current_generation,
                    "from": source_idx,
                    "to": dest_idx,
                    "timestamp": time.time(),
                }
            )
            record.island_idx = dest_idx
            record.migration_history = history
            session.commit()

    def get_program_brief(self, program_id: str) -> Optional[dict[str, Any]]:
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
            if record is None:
                return None
            return {
                "score": (
                    session.scalar(
                        select(ProgramEvaluationRecord.combined_score).where(
                            ProgramEvaluationRecord.program_id == program_id
                        )
                    )
                ),
                "children_count": record.children_count,
                "generation": record.generation,
                "metadata": record.program_metadata or {},
                "complexity": (
                    session.scalar(
                        select(ProgramEvaluationRecord.complexity).where(
                            ProgramEvaluationRecord.program_id == program_id
                        )
                    )
                    or 0.0
                ),
            }

    def insert_program_copy_from_object(
        self,
        *,
        program: Program,
        island_idx: int,
        metadata_updates: Dict[str, Any],
        clear_copy_flag: bool,
    ) -> str:
        if self.read_only:
            raise PermissionError("Cannot insert program copy in read-only mode.")
        metadata = dict(program.metadata or {})
        if clear_copy_flag:
            metadata.pop("_needs_island_copies", None)
        metadata.update(metadata_updates)
        new_id = str(uuid.uuid4())
        with self._session() as session:
            session.add(
                ProgramRecord(
                    id=new_id,
                    name=getattr(program, "name", None),
                    code=program.code,
                    language=program.language,
                    parent_id=program.parent_id,
                    generation=program.generation,
                    timestamp=program.timestamp,
                    children_count=program.children_count,
                    program_metadata=_clean_nan_values(metadata),
                    island_idx=island_idx,
                    migration_history=_clean_nan_values(program.migration_history or []),
                    system_prompt_id=program.system_prompt_id,
                )
            )
            session.flush()
            diagnostics = {}
            if program.metadata and program.metadata.get("code_analysis_metrics"):
                diagnostics["code_analysis_metrics"] = _clean_nan_values(
                    program.metadata.get("code_analysis_metrics")
                )
            session.add(
                ProgramEvaluationRecord(
                    id=str(uuid.uuid4()),
                    program_id=new_id,
                    correct=bool(program.correct),
                    combined_score=program.combined_score,
                    text_feedback=_normalize_text_feedback(program.text_feedback),
                    complexity=program.complexity,
                    compute_time_seconds=(
                        float((program.metadata or {}).get("compute_time"))
                        if (program.metadata or {}).get("compute_time") is not None
                        else None
                    ),
                    public_metrics_json=_clean_nan_values(program.public_metrics or {}),
                    private_metrics_json=_clean_nan_values(program.private_metrics or {}),
                    diagnostics_json=_clean_nan_values(diagnostics),
                )
            )
            session.add(
                ProgramProposalRecord(
                    id=str(uuid.uuid4()),
                    program_id=new_id,
                    patch_type=(program.metadata or {}).get("patch_type"),
                    patch_name=(program.metadata or {}).get("patch_name"),
                    code_diff=program.code_diff,
                    system_prompt_id=program.system_prompt_id,
                    proposal_metadata_json=_clean_nan_values(
                        {
                            key: value
                            for key, value in (program.metadata or {}).items()
                            if key not in {"code_analysis_metrics", "compute_time"}
                        }
                    ),
                )
            )
            if program.embedding:
                embedding_id = str(uuid.uuid4())
                session.add(
                    ProgramEmbeddingRecord(
                        id=embedding_id,
                        program_id=new_id,
                        vector_json=_clean_nan_values(program.embedding or []),
                        model_name=(program.metadata or {}).get("embedding_model"),
                        embedding_metadata_json={},
                    )
                )
                if program.embedding_pca_2d:
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=new_id,
                            embedding_id=embedding_id,
                            kind="pca_2d",
                            coords_json=_clean_nan_values(program.embedding_pca_2d or []),
                            cluster_id=program.embedding_cluster_id,
                            projection_metadata_json={},
                        )
                    )
                if program.embedding_pca_3d:
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=new_id,
                            embedding_id=embedding_id,
                            kind="pca_3d",
                            coords_json=_clean_nan_values(program.embedding_pca_3d or []),
                            cluster_id=program.embedding_cluster_id,
                            projection_metadata_json={},
                        )
                    )
            if self.inspiration_repo is None:
                raise ConnectionError("Repository inspiration store not initialized.")
            self.inspiration_repo.replace_for_child(
                new_id,
                [
                    InspirationUse(
                        child_program_id=new_id,
                        source_program_id=inspiration.source_program_id,
                        role=inspiration.role,
                        order_index=inspiration.order_index,
                        weight=inspiration.weight,
                        metadata=dict(inspiration.metadata or {}),
                    )
                    for inspiration in self._program_inspiration_uses(program)
                ],
                session=session,
            )
            session.commit()
        return new_id

    def insert_program_copy_from_row(
        self,
        *,
        source_program: Dict[str, Any],
        new_island_idx: int,
        new_parent_id: Optional[str],
        strategy: str,
        is_root: bool = False,
    ) -> str:
        if self.read_only:
            raise PermissionError("Cannot insert program copy in read-only mode.")
        raw_metadata = source_program.get("metadata") or {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata["_spawned_island"] = True
        metadata["_spawned_from_program_id"] = source_program["id"]
        metadata["_spawn_island_idx"] = new_island_idx
        metadata["_spawn_strategy"] = strategy
        if not is_root:
            metadata["_spawned_as_child"] = True
        new_id = str(uuid.uuid4())
        with self._session() as session:
            session.add(
                ProgramRecord(
                    id=new_id,
                    name=source_program.get("name"),
                    code=source_program["code"],
                    language=source_program["language"],
                    parent_id=new_parent_id,
                    generation=source_program.get("generation", 0),
                    timestamp=time.time(),
                    children_count=0,
                    program_metadata=_clean_nan_values(metadata),
                    island_idx=new_island_idx,
                    migration_history=_clean_nan_values(source_program.get("migration_history") or []),
                    system_prompt_id=source_program.get("system_prompt_id"),
                )
            )
            session.flush()
            diagnostics = {}
            if metadata.get("code_analysis_metrics"):
                diagnostics["code_analysis_metrics"] = _clean_nan_values(
                    metadata.get("code_analysis_metrics")
                )
            session.add(
                ProgramEvaluationRecord(
                    id=str(uuid.uuid4()),
                    program_id=new_id,
                    correct=bool(source_program.get("correct", 0)),
                    combined_score=source_program.get("combined_score"),
                    text_feedback=_normalize_text_feedback(source_program.get("text_feedback")),
                    complexity=float(source_program.get("complexity") or 0.0),
                    compute_time_seconds=(
                        float(metadata.get("compute_time"))
                        if metadata.get("compute_time") is not None
                        else None
                    ),
                    public_metrics_json=_clean_nan_values(source_program.get("public_metrics") or {}),
                    private_metrics_json=_clean_nan_values(source_program.get("private_metrics") or {}),
                    diagnostics_json=_clean_nan_values(diagnostics),
                )
            )
            session.add(
                ProgramProposalRecord(
                    id=str(uuid.uuid4()),
                    program_id=new_id,
                    patch_type=metadata.get("patch_type"),
                    patch_name=metadata.get("patch_name"),
                    code_diff=source_program.get("code_diff"),
                    system_prompt_id=source_program.get("system_prompt_id"),
                    proposal_metadata_json=_clean_nan_values(
                        {
                            key: value
                            for key, value in metadata.items()
                            if key not in {"code_analysis_metrics", "compute_time"}
                        }
                    ),
                )
            )
            source_embedding = list(source_program.get("embedding") or [])
            if source_embedding:
                embedding_id = str(uuid.uuid4())
                session.add(
                    ProgramEmbeddingRecord(
                        id=embedding_id,
                        program_id=new_id,
                        vector_json=_clean_nan_values(source_embedding),
                        model_name=metadata.get("embedding_model"),
                        embedding_metadata_json={},
                    )
                )
                if source_program.get("embedding_pca_2d"):
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=new_id,
                            embedding_id=embedding_id,
                            kind="pca_2d",
                            coords_json=_clean_nan_values(source_program.get("embedding_pca_2d") or []),
                            cluster_id=source_program.get("embedding_cluster_id"),
                            projection_metadata_json={},
                        )
                    )
                if source_program.get("embedding_pca_3d"):
                    session.add(
                        ProgramEmbeddingProjectionRecord(
                            id=str(uuid.uuid4()),
                            program_id=new_id,
                            embedding_id=embedding_id,
                            kind="pca_3d",
                            coords_json=_clean_nan_values(source_program.get("embedding_pca_3d") or []),
                            cluster_id=source_program.get("embedding_cluster_id"),
                            projection_metadata_json={},
                        )
                    )
            if new_parent_id:
                session.execute(
                    update(ProgramRecord)
                    .where(ProgramRecord.id == new_parent_id)
                    .values(children_count=ProgramRecord.children_count + 1)
                )
            if self.inspiration_repo is None:
                raise ConnectionError("Repository inspiration store not initialized.")
            source_archive_ids = list(
                source_program.get("archive_inspiration_ids") or []
            )
            source_top_k_ids = list(source_program.get("top_k_inspiration_ids") or [])
            self.inspiration_repo.replace_for_child(
                new_id,
                [
                    InspirationUse(
                        child_program_id=new_id,
                        source_program_id=source_program_id,
                        role="archive",
                        order_index=idx,
                    )
                    for idx, source_program_id in enumerate(source_archive_ids)
                ]
                + [
                    InspirationUse(
                        child_program_id=new_id,
                        source_program_id=source_program_id,
                        role="top_k",
                        order_index=idx,
                    )
                    for idx, source_program_id in enumerate(source_top_k_ids)
                ],
                session=session,
            )
            session.commit()
        return new_id

    def get_correct_child_rows(
        self,
        parent_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        query = (
            self._program_query(correct_only=True, parent_id=parent_id)
            .order_by(ProgramEvaluationRecord.combined_score.desc())
        )
        if limit is not None:
            query = query.limit(limit)
        with self._session() as session:
            records = session.execute(query).scalars().all()
        inspiration_index = self._build_inspiration_index([record.id for record in records])
        programs = [
            self._record_to_program(record, inspiration_index=inspiration_index)
            for record in records
        ]
        return [program.to_dict() for program in programs if program is not None]

    def list_embeddings_by_island(
        self,
        island_idx: int,
    ) -> List[tuple[str, list[float]]]:
        with self._session() as session:
            rows = session.execute(
                select(
                    ProgramEmbeddingRecord.program_id,
                    ProgramEmbeddingRecord.vector_json,
                )
                .join(ProgramRecord, ProgramRecord.id == ProgramEmbeddingRecord.program_id)
                .where(
                    ProgramRecord.island_idx == island_idx,
                    ProgramEmbeddingRecord.vector_json.is_not(None),
                )
            ).all()
        return [
            (str(program_id), list(embedding or []))
            for program_id, embedding in rows
            if embedding not in (None, [])
        ]

    def list_all_embeddings(self) -> List[tuple[str, list[float]]]:
        with self._session() as session:
            rows = session.execute(
                select(
                    ProgramEmbeddingRecord.program_id,
                    ProgramEmbeddingRecord.vector_json,
                ).where(
                    ProgramEmbeddingRecord.vector_json.is_not(None),
                )
            ).all()
        return [
            (str(program_id), list(embedding or []))
            for program_id, embedding in rows
            if embedding not in (None, [])
        ]

    def update_embedding_features(
        self,
        *,
        program_id: str,
        embedding_pca_2d: list[float],
        embedding_pca_3d: list[float],
        embedding_cluster_id: int,
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot update embedding features in read-only mode.")
        with self._session() as session:
            embedding_record = session.scalar(
                select(ProgramEmbeddingRecord).where(
                    ProgramEmbeddingRecord.program_id == program_id
                )
            )
            if embedding_record is None:
                embedding_record = ProgramEmbeddingRecord(
                    id=str(uuid.uuid4()),
                    program_id=program_id,
                    vector_json=[],
                    model_name=None,
                    embedding_metadata_json={},
                )
                session.add(embedding_record)
                session.flush()
            projection_2d = session.scalar(
                select(ProgramEmbeddingProjectionRecord).where(
                    ProgramEmbeddingProjectionRecord.program_id == program_id,
                    ProgramEmbeddingProjectionRecord.kind == "pca_2d",
                )
            )
            if projection_2d is None:
                projection_2d = ProgramEmbeddingProjectionRecord(
                    id=str(uuid.uuid4()),
                    program_id=program_id,
                    embedding_id=embedding_record.id,
                    kind="pca_2d",
                    coords_json=[],
                    cluster_id=None,
                    projection_metadata_json={},
                )
                session.add(projection_2d)
            projection_3d = session.scalar(
                select(ProgramEmbeddingProjectionRecord).where(
                    ProgramEmbeddingProjectionRecord.program_id == program_id,
                    ProgramEmbeddingProjectionRecord.kind == "pca_3d",
                )
            )
            if projection_3d is None:
                projection_3d = ProgramEmbeddingProjectionRecord(
                    id=str(uuid.uuid4()),
                    program_id=program_id,
                    embedding_id=embedding_record.id,
                    kind="pca_3d",
                    coords_json=[],
                    cluster_id=None,
                    projection_metadata_json={},
                )
                session.add(projection_3d)
            projection_2d.coords_json = _clean_nan_values(embedding_pca_2d)
            projection_2d.cluster_id = int(embedding_cluster_id)
            projection_3d.coords_json = _clean_nan_values(embedding_pca_3d)
            projection_3d.cluster_id = int(embedding_cluster_id)
            session.commit()

    def commit(self) -> None:
        if self.conn and not self.read_only:
            self.conn.commit()

    def update_program_metadata(
        self,
        program_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot update program metadata in read-only mode.")
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
            if record is None:
                return
            record.program_metadata = _clean_nan_values(metadata)
            session.commit()

    def get_best(
        self,
        metric: Optional[str] = None,
        *,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self.list_correct(island_idx=island_idx)
        if not programs:
            return None
        if metric:
            eligible = [p for p in programs if p.public_metrics and metric in p.public_metrics]
            ranked = sorted(
                eligible,
                key=lambda p: p.public_metrics.get(metric, -float("inf")),
                reverse=True,
            )
        elif any(p.combined_score is not None for p in programs):
            ranked = sorted(
                [p for p in programs if p.combined_score is not None],
                key=lambda p: p.combined_score or -float("inf"),
                reverse=True,
            )
        else:
            ranked = sorted(
                [p for p in programs if p.public_metrics],
                key=lambda p: (
                    sum(p.public_metrics.values()) / len(p.public_metrics)
                    if p.public_metrics
                    else -float("inf")
                ),
                reverse=True,
            )
        if not ranked:
            return None
        best = ranked[0]
        if metric is None and island_idx is None and best.id != self.best_program_id:
            self.best_program_id = best.id
            if not self.read_only:
                self._update_metadata("best_program_id", best.id)
        return best

    def get_ancestry(self, program_id: str, *, max_ancestors: int = 10) -> List[Program]:
        ancestors: List[Program] = []
        current = self.get(program_id)
        for _ in range(max_ancestors):
            if current is None or not current.parent_id:
                break
            current = self.get(current.parent_id)
            if current is None:
                break
            ancestors.append(current)
        ancestors.reverse()
        return ancestors

    def list_all(self) -> List[Program]:
        return self._list_programs(
            self._program_query().order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_correct(self, *, island_idx: Optional[int] = None) -> List[Program]:
        return self._list_programs(
            self._program_query(correct_only=True, island_idx=island_idx).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_incorrect(self, *, island_idx: Optional[int] = None) -> List[Program]:
        return self._list_programs(
            self._program_query(correct_only=False, island_idx=island_idx).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_by_island(
        self,
        island_idx: int,
        *,
        correct_only: bool = False,
    ) -> List[Program]:
        return self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_initialized_islands(self) -> List[Island]:
        if self.island_repo is None:
            raise ConnectionError("Repository island view not initialized.")
        return self.island_repo.list_initialized_islands()

    def list_initialized_island_ids(self) -> List[int]:
        if self.island_repo is None:
            raise ConnectionError("Repository island view not initialized.")
        return self.island_repo.list_initialized_island_ids()

    def list_islands(self) -> List[Island]:
        if self.island_repo is None:
            raise ConnectionError("Repository island view not initialized.")
        return self.island_repo.list_islands()

    def get_island_program_counts(
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, int]:
        if not island_indices:
            return {}
        counts = {int(idx): 0 for idx in island_indices}
        with self._session() as session:
            rows = session.execute(
                select(ProgramRecord.island_idx, func.count())
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
                .where(
                    ProgramRecord.island_idx.in_(list(island_indices)),
                    ProgramEvaluationRecord.correct.is_(True),
                )
                .group_by(ProgramRecord.island_idx)
            ).all()
        for island_idx, count in rows:
            counts[int(island_idx)] = int(count)
        return counts

    def get_island_best_scores(
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, float]:
        if not island_indices:
            return {}
        with self._session() as session:
            rows = session.execute(
                select(
                    ProgramRecord.island_idx,
                    func.max(ProgramEvaluationRecord.combined_score),
                )
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
                .where(
                    ProgramRecord.island_idx.in_(list(island_indices)),
                    ProgramEvaluationRecord.correct.is_(True),
                )
                .group_by(ProgramRecord.island_idx)
            ).all()
        return {
            int(island_idx): float(score) if score is not None else 0.0
            for island_idx, score in rows
        }

    def get_earliest(
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            ).limit(1)
        )
        return programs[0] if programs else None

    def get_most_recent(
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.desc(),
                ProgramRecord.timestamp.desc(),
                ProgramRecord.id.desc(),
            ).limit(1)
        )
        return programs[0] if programs else None

    def list_by_generation(self, generation: int) -> List[Program]:
        return self._list_programs(
            self._program_query(generation=generation).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_top(
        self,
        *,
        n: int = 10,
        metric: Optional[str] = "combined_score",
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> List[Program]:
        base = self._program_query(
            correct_only=True if correct_only else None,
            island_idx=island_idx,
        )
        if metric == "combined_score":
            return self._list_programs(
                base.where(ProgramEvaluationRecord.combined_score.is_not(None))
                .order_by(ProgramEvaluationRecord.combined_score.desc())
                .limit(n)
            )
        if metric == "timestamp":
            return self._list_programs(base.order_by(ProgramRecord.timestamp.desc()).limit(n))
        programs = self._list_programs(base)
        if not programs:
            return []
        if metric:
            ranked = sorted(
                [p for p in programs if p.public_metrics and metric in p.public_metrics],
                key=lambda p: p.public_metrics.get(metric, -float("inf")),
                reverse=True,
            )
        else:
            ranked = sorted(
                [p for p in programs if p.public_metrics],
                key=lambda p: (
                    sum(p.public_metrics.values()) / len(p.public_metrics)
                    if p.public_metrics
                    else -float("inf")
                ),
                reverse=True,
            )
        return ranked[:n]

    def get_summaries(self) -> List[dict[str, Any]]:
        with self._session() as session:
            records = session.execute(select(ProgramRecord)).scalars().all()
        program_ids = [record.id for record in records]
        inspiration_index = self._build_inspiration_index(program_ids)
        evaluation_index = self._build_evaluation_index(program_ids)
        proposal_index = self._build_proposal_index(program_ids)
        projection_index = self._build_projection_index(program_ids)
        return [
            self._record_to_summary(
                record,
                inspiration_index=inspiration_index,
                evaluation_index=evaluation_index,
                proposal_index=proposal_index,
                projection_index=projection_index,
            )
            for record in records
        ]

    def get_count_snapshot(self) -> ProgramCountSnapshot:
        with self._session() as session:
            count, max_timestamp = session.execute(
                select(func.count(ProgramRecord.id), func.max(ProgramRecord.timestamp))
            ).one()
        return ProgramCountSnapshot(count=int(count or 0), max_timestamp=max_timestamp)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
