from __future__ import annotations

import logging
from dataclasses import dataclass

from shinka.controllers.database_controller import DatabaseController
from .connector import DatabaseConnector

logger = logging.getLogger(__name__)


@dataclass
class RepositoryBundle:
    """
    Storage/bootstrap bundle for repository-backed runtime state.

    This owns:
    - opening the sqlite connection
    - handling read-only vs read-write mode
    - lightweight startup recovery checks
    - constructing repository objects on top of that connection

    It is the replacement path for the connection/bootstrap logic that still
    lives inside `ProgramDatabase.__init__`.
    """

    db_path: str | None
    num_islands: int
    read_only: bool
    connector: DatabaseConnector
    controller: DatabaseController

    @classmethod
    def open(
        cls,
        *,
        db_path: str | None = None,
        num_islands: int = 2,
        read_only: bool = False,
    ) -> "RepositoryBundle":
        connector = DatabaseConnector.open(
            db_path=db_path,
            num_islands=num_islands,
            read_only=read_only,
        )
        controller = DatabaseController(connector)
        return cls(
            db_path=db_path,
            num_islands=num_islands,
            read_only=read_only,
            connector=connector,
            controller=controller,
        )

    @property
    def metadata(self):
        return self.controller.metadata

    @property
    def islands(self):
        return self.controller.islands

    @property
    def inspirations(self):
        return self.controller.inspirations

    @property
    def programs(self):
        return self.controller.programs

    def close(self) -> None:
        self.controller.close()
