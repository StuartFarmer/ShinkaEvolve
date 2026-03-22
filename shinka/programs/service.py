from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy.orm import Session

from . import writes as program_writes
from .model import Program
from shinka.database import island_ops
from shinka.database.connection import Database

if TYPE_CHECKING:
    from .model import Program


@dataclass(frozen=True)
class ProgramWriteResult:
    program_id: str
    last_iteration: int
    spawned_island: bool = False
    ran_migration: bool = False


class ProgramWriteService:
    """
    Write-side lifecycle coordinator for newly produced programs.

    This is not the search controller. It owns only the consequences of adding a
    program to the system:

    - assign island
    - persist program through the program controller
    - run best/generation tracking callbacks
    - run island-copy / spawn / migration side effects
    - run optional post-write hooks like embedding recomputation or summaries
    """

    def __init__(
        self,
        *,
        db: Database,
        num_islands: int,
        migration_interval: int,
        migration_rate: float,
        island_elitism: bool,
        update_best_program: Callable[[Session, "Program"], None],
        recompute_embeddings: Optional[Callable[[], None]] = None,
        print_program_summary: Optional[Callable[["Program"], None]] = None,
        maybe_spawn_island: Optional[Callable[[Session, int], bool]] = None,
    ) -> None:
        self.db = db
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.update_best_program = update_best_program
        self.recompute_embeddings = recompute_embeddings
        self.print_program_summary = print_program_summary
        self.maybe_spawn_island = maybe_spawn_island

    def add(
        self,
        program: "Program",
        *,
        verbose: bool = False,
        current_last_iteration: int = 0,
    ) -> ProgramWriteResult:
        with self.db.session_scope() as session:
            island_ops.assign_program(session, program, num_islands=self.num_islands)
            program_id = program_writes.add_program(session, program, verbose=False)
            self.update_best_program(session, program)

            last_iteration = max(current_last_iteration, int(program.generation))

            if bool(program.metadata and program.metadata.get("_needs_island_copies")):
                island_ops.copy_program_to_islands(
                    session,
                    program,
                    num_islands=self.num_islands,
                )
                if program.metadata:
                    program.metadata.pop("_needs_island_copies", None)
                    program_writes.update_program_metadata(session, program.id, program.metadata)

            spawned_island = False
            if self.maybe_spawn_island is not None:
                spawned_island = bool(self.maybe_spawn_island(session, program.generation))

            ran_migration = False
            if (
                program.generation > 0
                and self.migration_interval > 0
                and (program.generation % self.migration_interval == 0)
            ):
                island_ops.perform_migration(
                    session,
                    num_islands=self.num_islands,
                    migration_rate=self.migration_rate,
                    island_elitism=self.island_elitism,
                    current_generation=last_iteration,
                )
                ran_migration = True

        if self.recompute_embeddings is not None:
            self.recompute_embeddings()

        if verbose and self.print_program_summary is not None:
            self.print_program_summary(program)

        return ProgramWriteResult(
            program_id=program_id,
            last_iteration=last_iteration,
            spawned_island=spawned_island,
            ran_migration=ran_migration,
        )
