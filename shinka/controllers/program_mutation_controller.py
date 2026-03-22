from __future__ import annotations

import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from shinka.database.program import Program
from shinka.database.models import (
    ProgramEmbeddingProjectionRecord,
    ProgramEmbeddingRecord,
    ProgramEvaluationRecord,
    ProgramProposalRecord,
    ProgramRecord,
)
from shinka.database.complexity import analyze_code_metrics

from .inspiration_controller import InspirationController
from .program_query_controller import ProgramQueryController
from .run_state_controller import RunStateController
from .types import InspirationUse

if TYPE_CHECKING:
    from .database_controller import DatabaseController

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


class ProgramMutationController:
    """Write/mutation layer for programs and program-related persistence helpers."""

    def __init__(self, database: "DatabaseController") -> None:
        self.database = database
        self._session_factory = database.SessionLocal
        self.read_only = database.read_only
        self.run_state = RunStateController(database)
        self.inspirations = InspirationController(database)
        self.query = ProgramQueryController(database)

    def _session(self) -> Session:
        return self._session_factory()

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
            self.inspirations.replace_for_child(
                program.id,
                self._program_inspiration_uses(program),
                session=session,
            )
            session.commit()

        snapshot = self.run_state.load_snapshot()
        if program.generation > snapshot.last_iteration:
            self.run_state.set("last_iteration", str(program.generation))

        current_best = self.query.get_best()
        if current_best is not None:
            self.run_state.set("best_program_id", current_best.id)

        if verbose:
            logger.info(
                "Program %s added to store - score: %s.",
                program.id,
                program.combined_score,
            )
        return program.id

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
                    ProgramEvaluationRecord.correct.is_(True),
                )
            )
            if island_elitism:
                query = base.order_by(ProgramEvaluationRecord.combined_score.desc()).limit(
                    num_migrants
                )
                return [str(program_id) for (program_id,) in session.execute(query).all()]

            rows = session.execute(base).all()
            ids = [str(program_id) for (program_id,) in rows]
            if len(ids) <= num_migrants:
                return ids
            return ids[:num_migrants]

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
            self.inspirations.replace_for_child(
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
            source_archive_ids = list(source_program.get("archive_inspiration_ids") or [])
            source_top_k_ids = list(source_program.get("top_k_inspiration_ids") or [])
            self.inspirations.replace_for_child(
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

    def commit(self) -> None:
        if self.database.conn and not self.read_only:
            self.database.conn.commit()
