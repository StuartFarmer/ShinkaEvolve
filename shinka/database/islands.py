import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import rich  # type: ignore
import rich.box  # type: ignore
from rich.console import Console as RichConsole  # type: ignore
from rich.table import Table as RichTable  # type: ignore

from shinka.controllers.island_controller import IslandController

if TYPE_CHECKING:
    from .archive_policy import ArchivePolicy
    from .repository import ProgramRepository

logger = logging.getLogger(__name__)


class CombinedIslandManager:
    """
    Island operations for program search.

    Islands are not first-class rows. They are computed from programs grouped by
    `island_idx`. This manager keeps the operational surface small:

    - assign a new program to an island
    - copy seed programs across islands
    - migrate programs by updating `island_idx`
    - spawn a new island from an existing source program/subtree
    """

    def __init__(
        self,
        *,
        num_islands: int,
        migration_interval: int,
        migration_rate: float,
        island_elitism: bool,
        island_spawn_strategy: str,
        island_spawn_subtree_size: int,
        program_repository: "ProgramRepository",
        island_controller: IslandController,
        archive_policy: "ArchivePolicy",
    ):
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.island_spawn_strategy = island_spawn_strategy
        self.island_spawn_subtree_size = island_spawn_subtree_size
        self.program_repository = program_repository
        self.repository = island_controller
        self.archive_policy = archive_policy

    def assign_island(self, program: Any) -> None:
        if program.island_idx is not None:
            logger.debug(
                "Preserving explicitly assigned island %s for program %s",
                program.island_idx,
                program.id,
            )
            return

        if self.num_islands <= 0:
            program.island_idx = 0
            return

        if self._is_first_program():
            program.island_idx = 0
            if program.metadata is None:
                program.metadata = {}
            program.metadata["_needs_island_copies"] = True
            logger.debug(
                "Assigned first program %s to island 0 and marked for copying",
                program.id,
            )
            return

        if program.parent_id:
            parent_island = self.repository.get_program_island(program.parent_id)
            if parent_island is not None:
                program.island_idx = parent_island
                logger.debug(
                    "Assigned program %s to parent's island %s",
                    program.id,
                    parent_island,
                )
                return

        initialized_islands = set(self.get_initialized_islands())
        uninitialized = [i for i in range(self.num_islands) if i not in initialized_islands]
        if uninitialized:
            program.island_idx = min(uninitialized)
            logger.debug(
                "Assigned program %s to first uninitialized island %s",
                program.id,
                program.island_idx,
            )
            return

        program.island_idx = random.randint(0, self.num_islands - 1)
        logger.debug(
            "Assigned program %s to random island %s",
            program.id,
            program.island_idx,
        )

    def perform_migration(self, current_generation: int) -> bool:
        if self.num_islands < 2 or self.migration_rate <= 0:
            return False

        logger.info("Performing island migration at generation %s", current_generation)

        migrations_summary: Dict[int, Dict[int, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        migrated_ids: set[str] = set()

        for source_idx in range(self.num_islands):
            island_size = self._count_island_programs(source_idx)
            if island_size <= 1:
                continue

            num_migrants = max(1, int(island_size * self.migration_rate))
            dest_islands = [idx for idx in range(self.num_islands) if idx != source_idx]
            if not dest_islands:
                continue

            for migrant_id in self._select_migrants(
                source_idx=source_idx,
                num_migrants=num_migrants,
                island_elitism=self.island_elitism,
            ):
                if migrant_id in migrated_ids:
                    logger.warning(
                        "Program %s already selected for migration, skipping duplicate",
                        migrant_id[:8] + "...",
                    )
                    continue
                migrated_ids.add(migrant_id)
                dest_idx = random.choice(dest_islands)
                self._migrate_program(
                    migrant_id=migrant_id,
                    source_idx=source_idx,
                    dest_idx=dest_idx,
                    current_generation=current_generation,
                )
                migrations_summary[source_idx][dest_idx].append(migrant_id)

        self.program_repository.commit()

        if migrations_summary:
            self._print_migration_summary(migrations_summary)

        total_migrated = sum(
            len(programs)
            for per_source in migrations_summary.values()
            for programs in per_source.values()
        )
        logger.info("Migration complete. Migrated %s programs.", total_migrated)
        return total_migrated > 0

    def get_island_idx(self, program_id: str) -> Optional[int]:
        return self.repository.get_program_island(program_id)

    def get_initialized_islands(self) -> List[int]:
        return self.repository.list_initialized_island_ids()

    def are_all_islands_initialized(self) -> bool:
        return self.repository.are_all_islands_initialized()

    def should_schedule_migration(self, program: Any) -> bool:
        return (
            program.generation > 0
            and self.migration_interval > 0
            and (program.generation % self.migration_interval == 0)
        )

    def get_island_populations(self) -> Dict[int, int]:
        return self.repository.get_island_populations()

    def get_migration_info(self) -> Optional[str]:
        if self.migration_interval <= 0:
            return None
        migration_str = (
            f"{self.migration_interval}G, "
            f"{self.migration_rate * 100:.0f}%"
        )
        if self.island_elitism:
            migration_str += "(E)"
        return migration_str

    def format_island_display(self) -> str:
        populations = self.get_island_populations()
        if not populations:
            return f"0 programs in {self.num_islands} islands"

        parts = []
        for island_idx, count in sorted(populations.items()):
            island_color = f"color({30 + island_idx % 220})"
            parts.append(f"[{island_color}]I{island_idx}: {count}[/{island_color}]")
        return " | ".join(parts)

    def copy_program_to_islands(self, program: Any) -> List[str]:
        if self.num_islands <= 1:
            return []

        created_ids: List[str] = []
        for island_idx in range(1, self.num_islands):
            new_id = self.program_repository.insert_program_copy_from_object(
                program=program,
                island_idx=island_idx,
                metadata_updates={
                    "_is_island_copy": True,
                    "_original_program_id": program.id,
                },
                clear_copy_flag=True,
            )
            created_ids.append(new_id)
            logger.info(
                "Created copy %s of program %s for island %s",
                new_id[:8] + "...",
                program.id[:8] + "...",
                island_idx,
            )

        self.program_repository.commit()
        logger.info(
            "Created %s copies of program %s for islands 1-%s",
            len(created_ids),
            program.id[:8] + "...",
            self.num_islands - 1,
        )
        return created_ids

    def spawn_new_island(self) -> bool:
        source_program = self._get_spawn_source_program(self.island_spawn_strategy)
        if not source_program:
            logger.warning(
                "Cannot spawn island: no source program found for strategy '%s'",
                self.island_spawn_strategy,
            )
            return False

        new_island_idx = self.program_repository.get_next_island_index()
        programs_to_copy = self._collect_subtree_programs(
            source_program,
            self.island_spawn_subtree_size,
        )
        old_to_new_id: Dict[str, str] = {}

        for idx, prog in enumerate(programs_to_copy):
            is_root = idx == 0
            old_parent_id = prog.get("parent_id")
            if is_root:
                new_parent_id = None
            elif old_parent_id and old_parent_id in old_to_new_id:
                new_parent_id = old_to_new_id[old_parent_id]
            else:
                new_parent_id = None

            new_id = self.program_repository.insert_program_copy_from_row(
                source_program=prog,
                new_island_idx=new_island_idx,
                new_parent_id=new_parent_id,
                strategy=self.island_spawn_strategy,
                is_root=is_root,
            )
            old_to_new_id[prog["id"]] = new_id

        self.program_repository.commit()

        source_id = source_program["id"][:8] + "..."
        if len(programs_to_copy) == 1:
            logger.info(
                "🏝️ Spawned new island %s with program %s (copy of %s source %s)",
                new_island_idx,
                old_to_new_id[source_program["id"]][:8] + "...",
                self.island_spawn_strategy,
                source_id,
            )
        else:
            logger.info(
                "🏝️ Spawned new island %s with %s programs (subtree from %s source %s)",
                new_island_idx,
                len(programs_to_copy),
                self.island_spawn_strategy,
                source_id,
            )
        return True

    def _is_first_program(self) -> bool:
        return self.repository.get_program_count() == 0

    def _count_island_programs(self, island_idx: int) -> int:
        return self.program_repository.count_by_island(island_idx)

    def _select_migrants(
        self,
        *,
        source_idx: int,
        num_migrants: int,
        island_elitism: bool,
    ) -> List[str]:
        migrants = self.program_repository.list_migrant_ids(
            source_idx=source_idx,
            num_migrants=num_migrants,
            island_elitism=island_elitism,
        )
        if not migrants:
            logger.debug(
                "No correct generation > 0 programs available for migration from island %s",
                source_idx,
            )
        return migrants

    def _migrate_program(
        self,
        *,
        migrant_id: str,
        source_idx: int,
        dest_idx: int,
        current_generation: int,
    ) -> None:
        self.program_repository.migrate_program(
            migrant_id=migrant_id,
            source_idx=source_idx,
            dest_idx=dest_idx,
            current_generation=current_generation,
        )

    def _print_migration_summary(
        self, migrations_summary: Dict[int, Dict[int, List[str]]]
    ) -> None:
        console = RichConsole()
        table = RichTable(
            title="[bold]Island Migration Summary[/bold]",
            box=rich.box.ROUNDED,
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            width=120,
        )
        table.add_column("Source", justify="center", style="cyan", width=8)
        table.add_column("Dest", justify="center", style="magenta", width=6)
        table.add_column("Program IDs", justify="left", style="green", width=15)
        table.add_column("Gen.", justify="center", style="yellow", width=10)
        table.add_column("Score", justify="right", style="yellow", width=8)
        table.add_column("Children", justify="right", style="blue", width=13)
        table.add_column("Patch Name", justify="left", style="white", width=30, overflow="ellipsis")
        table.add_column("Type", justify="left", style="cyan", width=8, overflow="ellipsis")
        table.add_column("Complexity", justify="right", style="red", width=9)

        for source, destinations in sorted(migrations_summary.items()):
            for dest, program_ids in sorted(destinations.items()):
                for program_id in program_ids:
                    result = self.program_repository.get_program_brief(program_id)
                    if not result:
                        continue
                    metadata = result["metadata"]
                    patch_name = metadata.get("patch_name", "N/A")
                    patch_type = metadata.get("patch_type", "N/A")
                    table.add_row(
                        f"I{source}",
                        f"I{dest}",
                        program_id[:8] + "...",
                        str(result["generation"] or 0),
                        f"{result['score']:.3f}" if result["score"] is not None else "N/A",
                        str(result["children_count"] or 0),
                        patch_name[:28] if patch_name != "N/A" else "N/A",
                        patch_type,
                        f"{result['complexity']:.1f}" if result["complexity"] else "N/A",
                    )
        console.print(table)

    def _get_spawn_source_program(self, strategy: str) -> Optional[Dict]:
        if strategy == "initial":
            return self.program_repository.get_initial_program_row()
        if strategy == "best":
            return self.program_repository.get_best_program_row()
        if strategy == "archive_random":
            program = self.archive_policy.pick_random(
                self.program_repository.list_correct()
            )
            return program.to_dict() if program is not None else None
        logger.warning(
            "Unknown island_spawn_strategy '%s', falling back to 'initial'",
            strategy,
        )
        return self.program_repository.get_initial_program_row()

    def _collect_subtree_programs(self, root_program: Dict, max_size: int) -> List[Dict]:
        if max_size <= 1:
            return [root_program]

        collected = [root_program]
        queue = [root_program]
        remaining = max_size - 1

        while queue and remaining > 0:
            current = queue.pop(0)
            children = self.program_repository.get_correct_child_rows(
                current["id"],
                limit=remaining,
            )
            for child in children:
                if remaining <= 0:
                    break
                collected.append(child)
                queue.append(child)
                remaining -= 1
        return collected
