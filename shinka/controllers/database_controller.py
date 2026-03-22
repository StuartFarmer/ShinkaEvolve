from __future__ import annotations

from shinka.database.connector import DatabaseConnector

from .embedding_controller import EmbeddingController
from .island_controller import IslandController
from .inspiration_controller import InspirationController
from .metadata_controller import MetadataController
from .program_controller import ProgramController
from .run_state_controller import RunStateController


class DatabaseController:
    """Master controller facade for model-scoped controllers."""

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self.programs = ProgramController(connector)
        self.metadata = MetadataController(connector)
        self.inspirations = InspirationController(connector)
        self.embeddings = EmbeddingController(connector)
        self.islands = IslandController(connector)
        self.run_state = RunStateController(connector)
