from __future__ import annotations

from typing import TYPE_CHECKING

from .embedding_controller import EmbeddingController
from .inspiration_controller import InspirationController
from .metadata_controller import MetadataController
from .program_mutation_controller import ProgramMutationController
from .program_query_controller import ProgramQueryController
from .run_state_controller import RunStateController

if TYPE_CHECKING:
    from .database_controller import DatabaseController


class ProgramController:
    """Public CRUD/query controller facade for programs."""

    def __init__(self, database: "DatabaseController") -> None:
        self.database = database
        self.num_islands = database.num_islands
        self.run_state = RunStateController(database)
        self.metadata = MetadataController(database)
        self.inspirations = InspirationController(database)
        self.embeddings = EmbeddingController(database)
        self.run_state_controller = self.run_state
        self.metadata_controller = self.metadata
        self.inspiration_controller = self.inspirations
        self.embedding_controller = self.embeddings
        self.query = ProgramQueryController(database)
        self.mutations = ProgramMutationController(database)
        snapshot = self.run_state.load_snapshot()
        self.last_iteration = snapshot.last_iteration
        self.best_program_id = snapshot.best_program_id

    def get_metadata(self, key: str, default=None):
        if key in RunStateController.SUPPORTED_KEYS:
            return self.run_state.get(key, default)
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value):
        if key in RunStateController.SUPPORTED_KEYS:
            self.run_state.set(key, value)
            if key == "last_iteration":
                self.last_iteration = 0 if value is None else int(value)
            elif key == "best_program_id":
                self.best_program_id = value
            return
        self.metadata.set(key, value)

    def add(self, program, *, verbose: bool = False):
        program_id = self.mutations.add(program, verbose=verbose)
        self.last_iteration = max(self.last_iteration, int(program.generation))
        best_program = self.query.get_best()
        self.best_program_id = None if best_program is None else best_program.id
        return program_id

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

    def close(self) -> None:
        self.database.close()

    def __getattr__(self, name):
        if hasattr(self.query, name):
            return getattr(self.query, name)
        if hasattr(self.mutations, name):
            return getattr(self.mutations, name)
        raise AttributeError(name)
