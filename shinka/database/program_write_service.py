from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .program import Program
    from .islands import CombinedIslandManager
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
        island_manager: "CombinedIslandManager",
        update_best_program: Callable[["Program"], None],
        update_metadata: Callable[[str, Optional[str]], None],
        recompute_embeddings: Optional[Callable[[], None]] = None,
        print_program_summary: Optional[Callable[["Program"], None]] = None,
        maybe_spawn_island: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self.programs = programs
        self.island_manager = island_manager
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
        self.island_manager.assign_island(program)
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
            self.island_manager.copy_program_to_islands(program)
            if program.metadata:
                program.metadata.pop("_needs_island_copies", None)
                self.programs.update_program_metadata(program.id, program.metadata)
                self.programs.commit()

        spawned_island = False
        if self.maybe_spawn_island is not None:
            spawned_island = bool(self.maybe_spawn_island(program.generation))

        ran_migration = False
        if self.island_manager.should_schedule_migration(program):
            self.island_manager.perform_migration(last_iteration)
            ran_migration = True

        return ProgramWriteResult(
            program_id=program_id,
            last_iteration=last_iteration,
            spawned_island=spawned_island,
            ran_migration=ran_migration,
        )
