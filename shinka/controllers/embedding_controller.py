from __future__ import annotations

import logging
import math
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from shinka.database.models import (
    ProgramEmbeddingProjectionRecord,
    ProgramEmbeddingRecord,
    ProgramRecord,
)
from shinka.embed import EmbeddingClient

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


class EmbeddingController:
    """Controller for program embeddings and derived projections."""

    def __init__(
        self,
        database: "DatabaseController",
        *,
        embedding_client_factory: Callable[[], Optional[EmbeddingClient]] | None = None,
    ) -> None:
        self.database = database
        self._session_factory = database.SessionLocal
        self.read_only = database.read_only
        self.embedding_client_factory = embedding_client_factory

    @contextmanager
    def _managed_session(self, session: Session | None = None):
        if session is not None:
            yield session
            return
        managed = self._session_factory()
        try:
            yield managed
        finally:
            managed.close()

    def list_by_island(self, island_idx: int) -> List[tuple[str, list[float]]]:
        with self._managed_session() as session:
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

    def list_all(self) -> List[tuple[str, list[float]]]:
        with self._managed_session() as session:
            rows = session.execute(
                select(
                    ProgramEmbeddingRecord.program_id,
                    ProgramEmbeddingRecord.vector_json,
                ).where(ProgramEmbeddingRecord.vector_json.is_not(None))
            ).all()
        return [
            (str(program_id), list(embedding or []))
            for program_id, embedding in rows
            if embedding not in (None, [])
        ]

    def recompute(self, num_clusters: int = 4) -> None:
        if self.read_only:
            return

        rows = self.list_all()
        if len(rows) < num_clusters:
            if rows:
                logger.info(
                    "Not enough programs with embeddings (%s) to perform clustering. Need at least %s.",
                    len(rows),
                    num_clusters,
                )
            return

        if self.embedding_client_factory is None:
            return

        embedding_client = self.embedding_client_factory()
        if embedding_client is None:
            return

        program_ids = [program_id for program_id, _ in rows]
        embeddings = [embedding for _, embedding in rows]

        try:
            logger.info(
                "Recomputing PCA-reduced embedding features for %s programs.",
                len(program_ids),
            )
            reduced_2d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=2
            )
            reduced_3d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=3
            )
            cluster_ids = embedding_client.get_embedding_clusters(
                embeddings, num_clusters=num_clusters
            )
        except Exception as e:
            logger.error("Failed to recompute embedding features: %s", e)
            return

        try:
            for i, program_id in enumerate(program_ids):
                self.update_features(
                    program_id=program_id,
                    embedding_pca_2d=reduced_2d[i].tolist(),
                    embedding_pca_3d=reduced_3d[i].tolist(),
                    embedding_cluster_id=int(cluster_ids[i]),
                )
            logger.info(
                "Successfully updated embedding features for %s programs.",
                len(program_ids),
            )
        except Exception as e:
            logger.error("Failed to update programs with new embedding features: %s", e)
            raise

    def update_features(
        self,
        *,
        program_id: str,
        embedding_pca_2d: list[float],
        embedding_pca_3d: list[float],
        embedding_cluster_id: int,
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot update embedding features in read-only mode.")
        with self._managed_session() as session:
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
