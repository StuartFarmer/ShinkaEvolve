from __future__ import annotations

import time
import uuid

from shinka.core.context_sampler import ContextSampler, SampledContext
from shinka.database import DatabaseConfig, Program, ProgramRepository
from shinka.database.archive_policy import FitnessArchivePolicy


def make_program(
    *,
    code: str = "print('hello')",
    generation: int = 0,
    score: float = 0.0,
    correct: bool = True,
    island_idx: int | None = 0,
    parent_id: str | None = None,
    timestamp: float | None = None,
) -> Program:
    return Program(
        id=str(uuid.uuid4()),
        code=code,
        language="python",
        generation=generation,
        combined_score=score,
        correct=correct,
        island_idx=island_idx,
        parent_id=parent_id,
        timestamp=time.time() if timestamp is None else timestamp,
        public_metrics={"score": score},
    )


def test_program_repository_roundtrip_and_queries(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    repo = ProgramRepository(DatabaseConfig(db_path=str(db_path), num_islands=2))

    root = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    child = make_program(
        generation=1,
        score=3.0,
        correct=True,
        island_idx=1,
        parent_id=root.id,
        timestamp=2.0,
    )
    incorrect = make_program(
        generation=2,
        score=0.5,
        correct=False,
        island_idx=1,
        timestamp=3.0,
    )

    repo.add(root)
    repo.add(child)
    repo.add(incorrect)

    loaded_child = repo.get(child.id)
    assert loaded_child is not None
    assert loaded_child.parent_id == root.id

    assert repo.get_children_count(root.id) == 1
    assert [program.id for program in repo.list_correct()] == [root.id, child.id]
    assert [program.id for program in repo.list_by_island(1, correct_only=True)] == [child.id]
    assert {program.id for program in repo.list_incorrect(island_idx=1)} == {incorrect.id}
    assert repo.get_best().id == child.id
    assert repo.get_best(island_idx=0).id == root.id
    assert repo.get_ancestry(child.id)[0].id == root.id
    assert [island.island_idx for island in repo.list_initialized_islands()] == [0, 1]
    assert repo.get_island_program_counts([0, 1]) == {0: 1, 1: 1}
    assert repo.get_island_best_scores([0, 1]) == {0: 1.0, 1: 3.0}

    repo.set_metadata("beam_search_parent_id", child.id)
    assert repo.get_metadata("beam_search_parent_id") == child.id

    repo.close()


def test_fitness_archive_policy_recomputes_from_programs(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    config = DatabaseConfig(
        db_path=str(db_path),
        num_islands=1,
        archive_size=2,
        archive_selection_strategy="fitness",
    )
    repo = ProgramRepository(config)

    p1 = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    p2 = make_program(generation=1, score=4.0, correct=True, island_idx=0, timestamp=2.0)
    p3 = make_program(generation=2, score=2.5, correct=True, island_idx=0, timestamp=3.0)
    p4 = make_program(generation=3, score=0.5, correct=False, island_idx=0, timestamp=4.0)

    for program in [p1, p2, p3, p4]:
        repo.add(program)

    archive = FitnessArchivePolicy(config).compute(repo.list_all())
    assert [program.id for program in archive] == [p2.id, p3.id]

    repo.close()


def test_context_sampler_uses_repository_backed_archive_and_parent_selection(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    config = DatabaseConfig(
        db_path=str(db_path),
        num_islands=1,
        archive_size=3,
        parent_selection_strategy="winner_take_all",
        num_archive_inspirations=2,
        num_top_k_inspirations=1,
        enforce_island_separation=True,
    )
    repo = ProgramRepository(config)

    p0 = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    p1 = make_program(generation=1, score=2.0, correct=True, island_idx=0, timestamp=2.0)
    p2 = make_program(generation=2, score=5.0, correct=True, island_idx=0, timestamp=3.0)
    p3 = make_program(generation=3, score=4.0, correct=True, island_idx=0, timestamp=4.0)

    for program in [p0, p1, p2, p3]:
        repo.add(program)

    sampled = ContextSampler(repo).sample(target_generation=4)

    assert isinstance(sampled, SampledContext)
    assert sampled.parent.id == p2.id
    archive_ids = {program.id for program in sampled.archive_inspirations}
    topk_ids = {program.id for program in sampled.top_k_inspirations}
    assert p2.id not in archive_ids
    assert p2.id not in topk_ids
    assert len(sampled.archive_inspirations) <= 2
    assert len(sampled.top_k_inspirations) <= 1

    repo.close()


def test_context_sampler_fix_mode_returns_incorrect_parent_with_ancestry(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    config = DatabaseConfig(
        db_path=str(db_path),
        num_islands=1,
        archive_size=3,
        num_archive_inspirations=2,
        num_top_k_inspirations=1,
    )
    repo = ProgramRepository(config)

    root = make_program(generation=0, score=0.0, correct=False, island_idx=0, timestamp=1.0)
    child = make_program(
        generation=1,
        score=0.0,
        correct=False,
        island_idx=0,
        parent_id=root.id,
        timestamp=2.0,
    )
    repo.add(root)
    repo.add(child)

    sampled = ContextSampler(repo).sample(target_generation=2, with_fix_mode=True)

    assert sampled.needs_fix is True
    assert sampled.parent.id in {root.id, child.id}
    if sampled.parent.id == child.id:
        assert [program.id for program in sampled.archive_inspirations] == [root.id]

    repo.close()
