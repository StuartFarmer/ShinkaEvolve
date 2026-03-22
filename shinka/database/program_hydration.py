from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import inspiration_ops
from .models import (
    ProgramEmbeddingProjectionRecord,
    ProgramEmbeddingRecord,
    ProgramEvaluationRecord,
    ProgramProposalRecord,
    ProgramRecord,
)
from .program import Program


def build_inspiration_index(
    session: Session,
    child_program_ids: Sequence[str],
) -> Dict[str, Dict[str, list[str]]]:
    uses_by_child = inspiration_ops.list_for_children(session, child_program_ids)
    index: Dict[str, Dict[str, list[str]]] = {
        child_id: {"archive": [], "top_k": [], "ancestor": []}
        for child_id in child_program_ids
    }
    for child_id, inspirations in uses_by_child.items():
        for inspiration in inspirations:
            role_bucket = index.setdefault(
                child_id,
                {"archive": [], "top_k": [], "ancestor": []},
            )
            role_bucket.setdefault(inspiration.role, []).append(inspiration.source_program_id)
    return index


def build_evaluation_index(
    session: Session,
    program_ids: Sequence[str],
) -> Dict[str, ProgramEvaluationRecord]:
    ids = [program_id for program_id in program_ids if program_id]
    if not ids:
        return {}
    records = session.execute(
        select(ProgramEvaluationRecord).where(ProgramEvaluationRecord.program_id.in_(ids))
    ).scalars().all()
    return {record.program_id: record for record in records}


def build_proposal_index(
    session: Session,
    program_ids: Sequence[str],
) -> Dict[str, ProgramProposalRecord]:
    ids = [program_id for program_id in program_ids if program_id]
    if not ids:
        return {}
    records = session.execute(
        select(ProgramProposalRecord).where(ProgramProposalRecord.program_id.in_(ids))
    ).scalars().all()
    return {record.program_id: record for record in records}


def build_embedding_index(
    session: Session,
    program_ids: Sequence[str],
) -> Dict[str, ProgramEmbeddingRecord]:
    ids = [program_id for program_id in program_ids if program_id]
    if not ids:
        return {}
    records = session.execute(
        select(ProgramEmbeddingRecord).where(ProgramEmbeddingRecord.program_id.in_(ids))
    ).scalars().all()
    return {record.program_id: record for record in records}


def build_projection_index(
    session: Session,
    program_ids: Sequence[str],
) -> Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]:
    ids = [program_id for program_id in program_ids if program_id]
    if not ids:
        return {}
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


def record_to_program(
    session: Session,
    record: ProgramRecord | None,
    *,
    inspiration_index: Optional[Dict[str, Dict[str, list[str]]]] = None,
    evaluation_index: Optional[Dict[str, ProgramEvaluationRecord]] = None,
    proposal_index: Optional[Dict[str, ProgramProposalRecord]] = None,
    embedding_index: Optional[Dict[str, ProgramEmbeddingRecord]] = None,
    projection_index: Optional[Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]] = None,
) -> Optional[Program]:
    if record is None:
        return None
    if evaluation_index is None:
        evaluation_index = build_evaluation_index(session, [record.id])
    if proposal_index is None:
        proposal_index = build_proposal_index(session, [record.id])
    if embedding_index is None:
        embedding_index = build_embedding_index(session, [record.id])
    if projection_index is None:
        projection_index = build_projection_index(session, [record.id])
    inspiration_lists = (
        inspiration_index.get(record.id, {})
        if inspiration_index is not None
        else build_inspiration_index(session, [record.id]).get(record.id, {})
    )
    evaluation = evaluation_index.get(record.id)
    proposal = proposal_index.get(record.id)
    embedding = embedding_index.get(record.id)
    projections = projection_index.get(record.id, {})
    projection_2d = projections.get("pca_2d")
    projection_3d = projections.get("pca_3d")
    diagnostics = dict(evaluation.diagnostics_json or {}) if evaluation is not None else {}
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


def record_to_summary(
    session: Session,
    record: ProgramRecord,
    *,
    inspiration_index: Optional[Dict[str, Dict[str, list[str]]]] = None,
    evaluation_index: Optional[Dict[str, ProgramEvaluationRecord]] = None,
    proposal_index: Optional[Dict[str, ProgramProposalRecord]] = None,
    projection_index: Optional[Dict[str, Dict[str, ProgramEmbeddingProjectionRecord]]] = None,
) -> dict[str, Any]:
    if evaluation_index is None:
        evaluation_index = build_evaluation_index(session, [record.id])
    if proposal_index is None:
        proposal_index = build_proposal_index(session, [record.id])
    if projection_index is None:
        projection_index = build_projection_index(session, [record.id])
    inspiration_lists = (
        inspiration_index.get(record.id, {})
        if inspiration_index is not None
        else build_inspiration_index(session, [record.id]).get(record.id, {})
    )
    evaluation = evaluation_index.get(record.id)
    proposal = proposal_index.get(record.id)
    projections = projection_index.get(record.id, {})
    projection_2d = projections.get("pca_2d")
    projection_3d = projections.get("pca_3d")
    diagnostics = dict(evaluation.diagnostics_json or {}) if evaluation is not None else {}
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
