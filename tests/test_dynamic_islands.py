"""Test dynamic island spawning on stagnation."""

import tempfile
from pathlib import Path

from shinka.database import Program
from shinka.database.archive_policy import create_archive_policy
from shinka.database.program_write_service import ProgramWriteService
from shinka.controllers import DatabaseController


class RuntimeHarness:
    def __init__(
        self,
        db_path: str,
        *,
        num_islands: int,
        enable_dynamic_islands: bool,
        stagnation_threshold: int,
        island_spawn_strategy: str = "initial",
    ) -> None:
        self.controller = DatabaseController.open(
            db_path=db_path,
            num_islands=num_islands,
            read_only=False,
        )
        self.programs = self.controller.programs
        self.metadata_repo = self.controller.run_state
        self.num_islands = num_islands
        self.enable_dynamic_islands = enable_dynamic_islands
        self.stagnation_threshold = stagnation_threshold
        self.island_spawn_strategy = island_spawn_strategy
        self.best_score_generation = 0
        self.best_score_ever = None
        self.last_iteration = 0
        self.best_program_id = None

        self.archive_policy = create_archive_policy(
            archive_selection_strategy="fitness",
            archive_size=40,
            archive_criteria={"combined_score": 1.0},
        )
        self.write_service = ProgramWriteService(
            programs=self.programs,
            islands=self.controller.islands,
            num_islands=num_islands,
            migration_interval=10,
            migration_rate=0.0,
            island_elitism=True,
            update_best_program=self._update_best_program,
            update_metadata=self.programs.set_metadata,
            recompute_embeddings=None,
            print_program_summary=None,
            maybe_spawn_island=self.check_and_spawn_island_if_stagnant,
        )

    def _update_best_program(self, program: Program) -> None:
        if not program.correct:
            return
        current_best = (
            self.programs.get(self.best_program_id)
            if self.best_program_id
            else None
        )
        current_best_score = (
            float(current_best.combined_score or 0.0) if current_best is not None else None
        )
        new_score = float(program.combined_score or 0.0)
        if current_best_score is None or new_score > current_best_score:
            self.best_program_id = program.id
            self.programs.set_metadata("best_program_id", program.id)
            if self.best_score_ever is None or new_score > self.best_score_ever:
                self.best_score_ever = new_score
                self.best_score_generation = program.generation
                self.programs.set_metadata(
                    "best_score_generation",
                    str(self.best_score_generation),
                )
                self.programs.set_metadata(
                    "best_score_ever",
                    str(self.best_score_ever),
                )

    def add(self, program: Program) -> None:
        result = self.write_service.add(
            program,
            verbose=False,
            current_last_iteration=self.last_iteration,
        )
        self.last_iteration = result.last_iteration

    def is_stagnant(self, current_generation: int) -> bool:
        if not self.enable_dynamic_islands:
            return False
        return current_generation - self.best_score_generation >= self.stagnation_threshold

    def check_and_spawn_island_if_stagnant(self, current_generation: int) -> bool:
        if not self.is_stagnant(current_generation):
            return False
        spawned = self.controller.islands.spawn_island(
            self.programs,
            self.archive_policy,
            strategy=self.island_spawn_strategy,
            subtree_size=1,
        )
        if spawned:
            self.best_score_generation = current_generation
            self.programs.set_metadata(
                "best_score_generation",
                str(self.best_score_generation),
            )
        return spawned

    def close(self) -> None:
        self.controller.close()


def test_stagnation_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_stagnation.db"
        runtime = RuntimeHarness(
            str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=5,
        )

        initial_program = Program(
            id="initial_prog",
            code="def initial(): return 0",
            correct=True,
            combined_score=1.0,
            generation=0,
            island_idx=0,
        )
        runtime.add(initial_program)

        assert not runtime.is_stagnant(0)
        assert not runtime.is_stagnant(4)
        assert runtime.is_stagnant(5)
        assert runtime.is_stagnant(10)
        runtime.close()


def test_dynamic_island_spawning():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_spawn.db"
        runtime = RuntimeHarness(
            str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=3,
        )

        runtime.add(
            Program(
                id="initial_prog",
                code="def initial(): return 0",
                correct=True,
                combined_score=1.0,
                generation=0,
                island_idx=0,
            )
        )
        initial_islands = runtime.controller.islands.get_island_populations()

        for gen in range(1, 5):
            runtime.add(
                Program(
                    id=f"prog_gen_{gen}",
                    code=f"def test(): return {gen}",
                    correct=True,
                    combined_score=0.5,
                    generation=gen,
                    island_idx=0,
                )
            )

        final_islands = runtime.controller.islands.get_island_populations()
        assert len(final_islands) > len(initial_islands)
        assert max(final_islands.keys()) >= runtime.num_islands
        runtime.close()


def test_no_spawning_when_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_disabled.db"
        runtime = RuntimeHarness(
            str(db_path),
            num_islands=2,
            enable_dynamic_islands=False,
            stagnation_threshold=3,
        )

        runtime.add(
            Program(
                id="initial_prog",
                code="def initial(): return 0",
                correct=True,
                combined_score=1.0,
                generation=0,
                island_idx=0,
            )
        )
        initial_islands = runtime.controller.islands.get_island_populations()

        for gen in range(1, 10):
            runtime.add(
                Program(
                    id=f"prog_gen_{gen}",
                    code=f"def test(): return {gen}",
                    correct=True,
                    combined_score=0.5,
                    generation=gen,
                    island_idx=0,
                )
            )

        final_islands = runtime.controller.islands.get_island_populations()
        assert len(final_islands) == len(initial_islands)
        runtime.close()


def test_stagnation_reset_on_improvement():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reset.db"
        runtime = RuntimeHarness(
            str(db_path),
            num_islands=2,
            enable_dynamic_islands=True,
            stagnation_threshold=5,
        )

        runtime.add(
            Program(
                id="initial_prog",
                code="def initial(): return 0",
                correct=True,
                combined_score=1.0,
                generation=0,
                island_idx=0,
            )
        )

        for gen in range(1, 4):
            runtime.add(
                Program(
                    id=f"prog_gen_{gen}",
                    code=f"def test(): return {gen}",
                    correct=True,
                    combined_score=0.5,
                    generation=gen,
                    island_idx=0,
                )
            )

        runtime.add(
            Program(
                id="better_prog",
                code="def better(): return 100",
                correct=True,
                combined_score=2.0,
                generation=4,
                island_idx=0,
            )
        )

        assert runtime.best_score_generation == 4
        assert runtime.best_score_ever == 2.0
        assert not runtime.is_stagnant(8)
        assert runtime.is_stagnant(9)
        runtime.close()


def test_spawn_strategies():
    strategies = ["initial", "best", "archive_random"]

    for strategy in strategies:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / f"test_spawn_{strategy}.db"
            runtime = RuntimeHarness(
                str(db_path),
                num_islands=2,
                enable_dynamic_islands=True,
                stagnation_threshold=3,
                island_spawn_strategy=strategy,
            )

            runtime.add(
                Program(
                    id="initial_prog",
                    code="def initial(): return 0",
                    correct=True,
                    combined_score=1.0,
                    generation=0,
                    island_idx=0,
                )
            )
            runtime.add(
                Program(
                    id="best_prog",
                    code="def best(): return 100",
                    correct=True,
                    combined_score=5.0,
                    generation=1,
                    island_idx=0,
                )
            )

            for gen in range(2, 6):
                runtime.add(
                    Program(
                        id=f"prog_gen_{gen}",
                        code=f"def test(): return {gen}",
                        correct=True,
                        combined_score=0.5,
                        generation=gen,
                        island_idx=0,
                    )
                )

            final_islands = runtime.controller.islands.get_island_populations()
            assert len(final_islands) > 2

            spawned_island_idx = max(final_islands.keys())
            spawned = runtime.programs.list_by_island(spawned_island_idx)
            assert spawned
            metadata = spawned[0].metadata or {}
            assert metadata.get("_spawn_strategy") == strategy
            runtime.close()


if __name__ == "__main__":
    test_stagnation_detection()
    test_dynamic_island_spawning()
