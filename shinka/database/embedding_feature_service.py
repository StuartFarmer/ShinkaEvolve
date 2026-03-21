from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from shinka.embed import EmbeddingClient

if TYPE_CHECKING:
    from .repository import ProgramRepository

logger = logging.getLogger(__name__)


class EmbeddingFeatureService:
    """
    Recomputes derived embedding features for already-persisted programs.

    This owns:
    - loading stored code embeddings from the repository
    - PCA-style dimensionality reduction
    - clustering
    - writing derived embedding features back through the repository
    """

    def __init__(
        self,
        repository: "ProgramRepository",
        *,
        embedding_client_factory: Callable[[], Optional[EmbeddingClient]],
        read_only: bool = False,
    ) -> None:
        self.repository = repository
        self.embedding_client_factory = embedding_client_factory
        self.read_only = read_only

    def recompute(self, num_clusters: int = 4) -> None:
        if self.read_only:
            return

        rows = self.repository.list_all_embeddings()
        if len(rows) < num_clusters:
            if rows:
                logger.info(
                    "Not enough programs with embeddings (%s) to perform clustering. Need at least %s.",
                    len(rows),
                    num_clusters,
                )
            return

        program_ids = [program_id for program_id, _ in rows]
        embeddings = [embedding for _, embedding in rows]
        embedding_client = self.embedding_client_factory()
        if embedding_client is None:
            return

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
                self.repository.update_embedding_features(
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
