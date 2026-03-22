from __future__ import annotations

import time

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class ProgramRecord(Base):
    __tablename__ = "programs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("projects.id"),
        nullable=True,
        index=True,
    )
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False, default="python")
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    children_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    program_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    island_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    migration_history: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    system_prompt_id: Mapped[str | None] = mapped_column(String, nullable=True)


class MetadataRecord(Base):
    __tablename__ = "metadata_store"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)


class ProgramInspirationRecord(Base):
    __tablename__ = "program_inspirations"
    __table_args__ = (
        UniqueConstraint(
            "child_program_id",
            "source_program_id",
            "role",
            "order_index",
            name="uq_program_inspiration_edge",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    child_program_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_program_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ProjectRecord(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    task_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)
    config_snapshot_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ProgramEvaluationRecord(Base):
    __tablename__ = "program_evaluations"
    __table_args__ = (
        UniqueConstraint("program_id", name="uq_program_evaluation_program"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    text_feedback: Mapped[str] = mapped_column(Text, nullable=False, default="")
    complexity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    compute_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    private_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    diagnostics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class ProgramProposalRecord(Base):
    __tablename__ = "program_proposals"
    __table_args__ = (
        UniqueConstraint("program_id", name="uq_program_proposal_program"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    patch_type: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    patch_name: Mapped[str | None] = mapped_column(String, nullable=True)
    code_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("prompts.id"),
        nullable=True,
    )
    proposal_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class ProgramEmbeddingRecord(Base):
    __tablename__ = "program_embeddings"
    __table_args__ = (
        UniqueConstraint("program_id", name="uq_program_embedding_program"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    vector_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)


class ProgramEmbeddingProjectionRecord(Base):
    __tablename__ = "program_embedding_projections"
    __table_args__ = (
        UniqueConstraint(
            "program_id",
            "kind",
            name="uq_program_embedding_projection_kind",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    embedding_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("program_embeddings.id"),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    coords_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    projection_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class PromptRecord(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    tag: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)
    prompt_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class PromptSnapshotRecord(Base):
    __tablename__ = "prompt_snapshots"
    __table_args__ = (
        UniqueConstraint("program_id", name="uq_prompt_snapshot_program"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    system_prompt_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("prompts.id"),
        nullable=True,
    )
    user_prompt_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("prompts.id"),
        nullable=True,
    )
    meta_prompt_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("prompts.id"),
        nullable=True,
    )
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)
    snapshot_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class ModelUsageRecord(Base):
    __tablename__ = "model_usages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    program_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    api_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    usage_metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class RunStateRecord(Base):
    __tablename__ = "run_state"

    id: Mapped[str] = mapped_column(String, primary_key=True, default="default")
    project_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("projects.id"),
        nullable=True,
        unique=True,
    )
    last_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_program_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=True,
    )
    beam_search_parent_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("programs.id"),
        nullable=True,
    )
    best_score_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_score_ever: Mapped[float | None] = mapped_column(Float, nullable=True)
    initial_program_count_adjustment: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
