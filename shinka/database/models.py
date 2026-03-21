from __future__ import annotations

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class ProgramRecord(Base):
    __tablename__ = "programs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False, default="python")
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    archive_inspiration_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    top_k_inspiration_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    code_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    private_metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    text_feedback: Mapped[str] = mapped_column(Text, nullable=False, default="")
    complexity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    embedding: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding_pca_2d: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding_pca_3d: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding_cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    children_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    program_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    island_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    migration_history: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    system_prompt_id: Mapped[str | None] = mapped_column(String, nullable=True)


class MetadataRecord(Base):
    __tablename__ = "metadata_store"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
