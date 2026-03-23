from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import reads as program_reads
from .model import Program
from shinka.database import inspiration_ops, run_state_ops
from shinka.database.complexity import analyze_code_metrics
from shinka.database.models import (
    ProgramEmbeddingProjectionRecord,
    ProgramEmbeddingRecord,
    ProgramEvaluationRecord,
    ProgramProposalRecord,
    ProgramRecord, ProgramInspirationRecord,
)
from shinka.database.types import InspirationUse

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


def _program_inspiration_uses(program: Program, *, child_program_id: Optional[str] = None) -> List[InspirationUse]:
    inspirations: List[InspirationUse] = []
    resolved_child_id = child_program_id or program.id
    for idx, source_program_id in enumerate(program.archive_inspiration_ids or []):
        inspirations.append(
            InspirationUse(
                child_program_id=resolved_child_id,
                source_program_id=source_program_id,
                role="archive",
                order_index=idx,
            )
        )
    for idx, source_program_id in enumerate(program.top_k_inspiration_ids or []):
        inspirations.append(
            InspirationUse(
                child_program_id=resolved_child_id,
                source_program_id=source_program_id,
                role="top_k",
                order_index=idx,
            )
        )
    return inspirations


def enrich_program_for_insert(program: Program) -> None:
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


def add_program(session: Session, program: Program, *, verbose: bool = False) -> str:
    enrich_program_for_insert(program)
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
    replace_for_child(
        session,
        program.id,
        _program_inspiration_uses(program),
    )

    snapshot = run_state_ops.load_snapshot(session, read_only=False)
    if program.generation > snapshot.last_iteration:
        run_state_ops.set(session, "last_iteration", str(program.generation))

    current_best = program_reads.get_best(session)
    if current_best is not None:
        run_state_ops.set(session, "best_program_id", current_best.id)

    if verbose:
        logger.info(
            "Program %s added to store - score: %s.",
            program.id,
            program.combined_score,
        )
    return program.id


def list_migrant_ids(
    session: Session,
    *,
    source_idx: int,
    num_migrants: int,
    island_elitism: bool,
) -> List[str]:
    base = (
        select(ProgramRecord.id)
        .join(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
        .where(
            ProgramRecord.island_idx == source_idx,
            ProgramEvaluationRecord.correct.is_(True),
        )
    )
    if island_elitism:
        query = base.order_by(ProgramEvaluationRecord.combined_score.desc()).limit(num_migrants)
        return [str(program_id) for (program_id,) in session.execute(query).all()]

    rows = session.execute(base).all()
    ids = [str(program_id) for (program_id,) in rows]
    if len(ids) <= num_migrants:
        return ids
    return ids[:num_migrants]


def migrate_program(
    session: Session,
    *,
    migrant_id: str,
    source_idx: int,
    dest_idx: int,
    current_generation: int,
) -> None:
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


def get_program_brief(session: Session, program_id: str) -> Optional[dict[str, Any]]:
    record = session.get(ProgramRecord, program_id)
    if record is None:
        return None
    return {
        "score": session.scalar(
            select(ProgramEvaluationRecord.combined_score).where(
                ProgramEvaluationRecord.program_id == program_id
            )
        ),
        "children_count": record.children_count,
        "generation": record.generation,
        "metadata": record.program_metadata or {},
        "complexity": session.scalar(
            select(ProgramEvaluationRecord.complexity).where(
                ProgramEvaluationRecord.program_id == program_id
            )
        )
        or 0.0,
    }


def insert_program_copy_from_object(
    session: Session,
    *,
    program: Program,
    island_idx: int,
    metadata_updates: Dict[str, Any],
    clear_copy_flag: bool,
) -> str:
    metadata = dict(program.metadata or {})
    if clear_copy_flag:
        metadata.pop("_needs_island_copies", None)
    metadata.update(metadata_updates)
    new_id = str(uuid.uuid4())
    copied = Program.from_dict({**program.to_dict(), "id": new_id, "island_idx": island_idx, "metadata": metadata})
    add_program(session, copied)
    return new_id


def insert_program_copy_from_row(
    session: Session,
    *,
    source_program: Dict[str, Any],
    new_island_idx: int,
    new_parent_id: Optional[str],
    strategy: str,
    is_root: bool = False,
) -> str:
    raw_metadata = source_program.get("metadata") or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["_spawned_island"] = True
    metadata["_spawned_from_program_id"] = source_program["id"]
    metadata["_spawn_island_idx"] = new_island_idx
    metadata["_spawn_strategy"] = strategy
    if not is_root:
        metadata["_spawned_as_child"] = True
    new_id = str(uuid.uuid4())
    copied = Program.from_dict(
        {
            **source_program,
            "id": new_id,
            "parent_id": new_parent_id,
            "timestamp": time.time(),
            "children_count": 0,
            "metadata": metadata,
            "island_idx": new_island_idx,
        }
    )
    add_program(session, copied)
    return new_id


def update_program_metadata(
    session: Session,
    program_id: str,
    metadata: Dict[str, Any],
) -> None:
    record = session.get(ProgramRecord, program_id)
    if record is None:
        return
    record.program_metadata = _clean_nan_values(metadata)


def replace_for_child(
    session: Session,
    child_program_id: str,
    inspirations: List[InspirationUse],
) -> None:
    session.execute(
        delete(ProgramInspirationRecord).where(
            ProgramInspirationRecord.child_program_id == child_program_id
        )
    )
    for inspiration in inspirations:
        session.add(
            ProgramInspirationRecord(
                id=str(uuid.uuid4()),
                child_program_id=inspiration.child_program_id,
                source_program_id=inspiration.source_program_id,
                role=inspiration.role,
                order_index=inspiration.order_index,
                weight=inspiration.weight,
                edge_metadata=dict(inspiration.metadata or {}),
            )
        )
