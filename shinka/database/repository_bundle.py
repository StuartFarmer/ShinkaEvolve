from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shinka.controllers.database_controller import DatabaseController
from .connector import DatabaseConnector

if TYPE_CHECKING:
    from .repository import ProgramRepository

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
    programs: "ProgramRepository"
    controller: DatabaseController

    @classmethod
    def open(
        cls,
        *,
        db_path: str | None = None,
        num_islands: int = 2,
        read_only: bool = False,
    ) -> "RepositoryBundle":
        from .repository import ProgramRepository

        connector = DatabaseConnector.open(
            db_path=db_path,
            num_islands=num_islands,
            read_only=read_only,
        )
        programs = ProgramRepository.from_existing_connection(
            db_path=db_path,
            num_islands=num_islands,
            conn=connector.conn,
            cursor=connector.cursor,
            read_only=read_only,
            ensure_schema=not read_only,
        )
        return cls(
            db_path=db_path,
            num_islands=num_islands,
            read_only=read_only,
            connector=connector,
            programs=programs,
            controller=DatabaseController(connector),
        )

    @property
    def metadata(self):
        return self.programs.metadata_repo

    @property
    def islands(self):
        return self.programs.island_repo

    def close(self) -> None:
        self.programs.close()
