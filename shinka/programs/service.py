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

def perform_migration(
    session: Session,
    *,
    num_islands: int,
    migration_rate: float,
    island_elitism: bool,
    current_generation: int,
) -> bool:
    if num_islands < 2 or migration_rate <= 0:
        return False

    migrated = 0
    migrated_ids: set[str] = set()
    for source_idx in range(num_islands):
        island_size = program_reads.count_by_island(session, source_idx)
        if island_size <= 1:
            continue
        num_migrants = max(1, int(island_size * migration_rate))
        dest_islands = [idx for idx in range(num_islands) if idx != source_idx]
        if not dest_islands:
            continue
        migrants = program_writes.list_migrant_ids(
            session,
            source_idx=source_idx,
            num_migrants=num_migrants,
            island_elitism=island_elitism,
        )
        for migrant_id in migrants:
            if migrant_id in migrated_ids:
                continue
            migrated_ids.add(migrant_id)
            program_writes.migrate_program(
                session,
                migrant_id=migrant_id,
                source_idx=source_idx,
                dest_idx=random.choice(dest_islands),
                current_generation=current_generation,
            )
            migrated += 1
    return migrated > 0


def copy_program_to_islands(
    session: Session,
    program: Program,
    *,
    num_islands: int,
) -> List[str]:
    if num_islands <= 1:
        return []
    created_ids: List[str] = []
    for island_idx in range(1, num_islands):
        created_ids.append(
            program_writes.insert_program_copy_from_object(
                session,
                program=program,
                island_idx=island_idx,
                metadata_updates={
                    "_is_island_copy": True,
                    "_original_program_id": program.id,
                },
                clear_copy_flag=True,
            )
        )
    return created_ids


def assign_program(session: Session, program: Any, *, num_islands: int) -> None:
    if program.island_idx is not None:
        return
    if num_islands <= 0:
        program.island_idx = 0
        return
    if get_program_count(session) == 0:
        program.island_idx = 0
        if program.metadata is None:
            program.metadata = {}
        program.metadata["_needs_island_copies"] = True
        return
    if program.parent_id:
        parent_island = get_program_island(session, program.parent_id)
        if parent_island is not None:
            program.island_idx = parent_island
            return
    initialized = set(list_initialized_island_ids(session, num_islands=num_islands))
    uninitialized = [idx for idx in range(num_islands) if idx not in initialized]
    if uninitialized:
        program.island_idx = min(uninitialized)
        return
    program.island_idx = random.randint(0, num_islands - 1)

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
            assign_program(session, program, num_islands=self.num_islands)
            program_id = program_writes.add_program(session, program, verbose=False)
            self.update_best_program(session, program)

            last_iteration = max(current_last_iteration, int(program.generation))

            if bool(program.metadata and program.metadata.get("_needs_island_copies")):
                copy_program_to_islands(
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
                perform_migration(
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
