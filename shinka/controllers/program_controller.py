from __future__ import annotations

from shinka.database.connector import DatabaseConnector
from .embedding_controller import EmbeddingController
from .inspiration_controller import InspirationController
from .metadata_controller import MetadataController
from .run_state_controller import RunStateController


class ProgramController:
    """Public CRUD/query controller facade for programs."""

    def __init__(self, connector: DatabaseConnector) -> None:
        from shinka.database.repository import ProgramRepository

        self.connector = connector
        self.run_state = RunStateController(connector)
        self.metadata = MetadataController(connector)
        self.inspirations = InspirationController(connector)
        self.embeddings = EmbeddingController(connector)
        self._store = ProgramRepository.from_existing_connection(
            db_path=connector.db_path,
            num_islands=connector.num_islands,
            conn=connector.conn,
            cursor=connector.cursor,
            read_only=connector.read_only,
            ensure_schema=not connector.read_only,
        )

    def get_metadata(self, key: str, default=None):
        if key in RunStateController.SUPPORTED_KEYS:
            return self.run_state.get(key, default)
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value):
        if key in RunStateController.SUPPORTED_KEYS:
            self.run_state.set(key, value)
            return
        self.metadata.set(key, value)

    def get_inspiration_uses(self, child_program_id: str, *, role=None):
        inspirations = self.inspirations.list_for_child(child_program_id)
        if role is not None:
            inspirations = [insp for insp in inspirations if insp.role == role]
        return inspirations

    def get_inspiration_source_ids(self, child_program_id: str, *, role=None):
        return self.inspirations.list_sources_for_child(child_program_id, role=role)

    def get_inspired_child_ids(self, source_program_id: str, *, role=None):
        return self.inspirations.list_children_for_source(source_program_id, role=role)

    def count_inspiration_usage_by_source(self, source_program_id: str, *, role=None):
        return self.inspirations.count_usage_by_source(source_program_id, role=role)

    def count_inspiration_usage_by_role(self, role: str) -> int:
        return self.inspirations.count_usage_by_role(role)

    def list_embeddings_by_island(self, island_idx: int):
        return self.embeddings.list_by_island(island_idx)

    def list_all_embeddings(self):
        return self.embeddings.list_all()

    def update_embedding_features(
        self,
        *,
        program_id: str,
        embedding_pca_2d,
        embedding_pca_3d,
        embedding_cluster_id: int,
    ):
        self.embeddings.update_features(
            program_id=program_id,
            embedding_pca_2d=embedding_pca_2d,
            embedding_pca_3d=embedding_pca_3d,
            embedding_cluster_id=embedding_cluster_id,
        )

    def __getattr__(self, name):
        return getattr(self._store, name)
