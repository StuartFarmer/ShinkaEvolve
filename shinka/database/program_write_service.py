from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .program import Program
    from shinka.controllers.island_controller import IslandController
    from shinka.controllers.program_controller import ProgramController


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
        programs: "ProgramController",
        islands: "IslandController",
        num_islands: int,
        migration_interval: int,
        migration_rate: float,
        island_elitism: bool,
        update_best_program: Callable[["Program"], None],
        update_metadata: Callable[[str, Optional[str]], None],
        recompute_embeddings: Optional[Callable[[], None]] = None,
        print_program_summary: Optional[Callable[["Program"], None]] = None,
        maybe_spawn_island: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self.programs = programs
        self.islands = islands
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.update_best_program = update_best_program
        self.update_metadata = update_metadata
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
        self.islands.assign_program(program, num_islands=self.num_islands)
        program_id = self.programs.add(program, verbose=False)

        self.update_best_program(program)

        if self.recompute_embeddings is not None:
            self.recompute_embeddings()

        last_iteration = current_last_iteration
        if program.generation > last_iteration:
            last_iteration = program.generation
            self.update_metadata("last_iteration", str(last_iteration))

        if verbose and self.print_program_summary is not None:
            self.print_program_summary(program)

        if bool(program.metadata and program.metadata.get("_needs_island_copies")):
            self.islands.copy_program_to_islands(
                self.programs,
                program,
                num_islands=self.num_islands,
            )
            if program.metadata:
                program.metadata.pop("_needs_island_copies", None)
                self.programs.update_program_metadata(program.id, program.metadata)
                self.programs.commit()

        spawned_island = False
        if self.maybe_spawn_island is not None:
            spawned_island = bool(self.maybe_spawn_island(program.generation))

        ran_migration = False
        if (
            program.generation > 0
            and self.migration_interval > 0
            and (program.generation % self.migration_interval == 0)
        ):
            self.islands.perform_migration(
                self.programs,
                num_islands=self.num_islands,
                migration_rate=self.migration_rate,
                island_elitism=self.island_elitism,
                current_generation=last_iteration,
            )
            ran_migration = True

        return ProgramWriteResult(
            program_id=program_id,
            last_iteration=last_iteration,
            spawned_island=spawned_island,
            ran_migration=ran_migration,
        )
