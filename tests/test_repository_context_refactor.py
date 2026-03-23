from __future__ import annotations

import time
import uuid

from shinka.runtime.context import ContextSampler, SampledContext
from shinka.database import (
    Database,
    InspirationUse,
    Program,
    inspiration_ops,
    island_ops,
    program_reads,
    program_writes,
    run_state_ops,
)
from shinka.programs.archive import FitnessArchivePolicy

def count_usage_by_role(session: Session, role: str) -> int:
    query = select(func.count()).select_from(ProgramInspirationRecord).where(
        ProgramInspirationRecord.role == role
    )
    return int(session.scalar(query) or 0)

def list_sources_for_child(
    session: Session,
    child_program_id: str,
    *,
    role: Optional[str] = None,
) -> List[str]:
    inspirations = list_for_child(session, child_program_id)
    if role is not None:
        inspirations = [insp for insp in inspirations if insp.role == role]
    return [insp.source_program_id for insp in inspirations]


def list_children_for_source(
    session: Session,
    source_program_id: str,
    *,
    role: Optional[str] = None,
) -> List[str]:
    query = select(ProgramInspirationRecord.child_program_id).where(
        ProgramInspirationRecord.source_program_id == source_program_id
    )
    if role is not None:
        query = query.where(ProgramInspirationRecord.role == role)
    rows = session.execute(query).all()
    return [str(child_program_id) for (child_program_id,) in rows]


def count_usage_by_source(
    session: Session,
    source_program_id: str,
    *,
    role: Optional[str] = None,
) -> int:
    query = select(func.count()).select_from(ProgramInspirationRecord).where(
        ProgramInspirationRecord.source_program_id == source_program_id
    )
    if role is not None:
        query = query.where(ProgramInspirationRecord.role == role)
    return int(session.scalar(query) or 0)


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
    db = Database.open(db_path=str(db_path), num_islands=2)

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

    with db.session_scope() as session:
        program_writes.add_program(session, root)
        program_writes.add_program(session, child)
        program_writes.add_program(session, incorrect)

    with db.session_scope() as session:
        loaded_child = program_reads.get(session, child.id)
        assert loaded_child is not None
        assert loaded_child.parent_id == root.id

        assert program_reads.get_children_count(session, root.id) == 1
        assert [program.id for program in program_reads.list_correct(session)] == [
            root.id,
            child.id,
        ]
        assert [
            program.id
            for program in program_reads.list_by_island(
                session,
                1,
                correct_only=True,
            )
        ] == [child.id]
        assert {
            program.id
            for program in program_reads.list_incorrect(session, island_idx=1)
        } == {incorrect.id}
        assert program_reads.get_best(session).id == child.id
        assert program_reads.get_best(session, island_idx=0).id == root.id
        assert program_reads.get_ancestry(session, child.id)[0].id == root.id
        assert [
            island.island_idx
            for island in island_ops.list_initialized_islands(
                session,
                num_islands=2,
            )
        ] == [0, 1]
        assert program_reads.get_island_program_counts(session, [0, 1]) == {0: 1, 1: 1}
        assert program_reads.get_island_best_scores(session, [0, 1]) == {
            0: 1.0,
            1: 3.0,
        }

        run_state_ops.set(session, "beam_search_parent_id", child.id)
        assert run_state_ops.get(session, "beam_search_parent_id") == child.id

    db.close()


def test_fitness_archive_policy_recomputes_from_programs(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    db = Database.open(db_path=str(db_path), num_islands=1)

    p1 = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    p2 = make_program(generation=1, score=4.0, correct=True, island_idx=0, timestamp=2.0)
    p3 = make_program(generation=2, score=2.5, correct=True, island_idx=0, timestamp=3.0)
    p4 = make_program(generation=3, score=0.5, correct=False, island_idx=0, timestamp=4.0)

    with db.session_scope() as session:
        for program in [p1, p2, p3, p4]:
            program_writes.add_program(session, program)

    with db.session() as session:
        archive = FitnessArchivePolicy(archive_size=2)(program_reads.list_all(session))
        assert [program.id for program in archive] == [p2.id, p3.id]

    db.close()


def test_program_repository_persists_inspirations_in_join_table(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    db = Database.open(db_path=str(db_path), num_islands=1)

    source_a = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    source_b = make_program(generation=1, score=2.0, correct=True, island_idx=0, timestamp=2.0)
    child = make_program(
        generation=2,
        score=3.0,
        correct=True,
        island_idx=0,
        parent_id=source_b.id,
        timestamp=3.0,
    )
    child.archive_inspiration_ids = [source_a.id]
    child.top_k_inspiration_ids = [source_b.id]

    with db.session_scope() as session:
        program_writes.add_program(session, source_a)
        program_writes.add_program(session, source_b)
        program_writes.add_program(session, child)

    with db.session() as session:
        loaded = program_reads.get(session, child.id)
        assert loaded is not None
        assert loaded.archive_inspiration_ids == [source_a.id]
        assert loaded.top_k_inspiration_ids == [source_b.id]

        uses = inspiration_ops.list_for_child(session, child.id)
        assert uses == [
            InspirationUse(
                child_program_id=child.id,
                source_program_id=source_a.id,
                role="archive",
                order_index=0,
                weight=None,
                metadata={},
            ),
            InspirationUse(
                child_program_id=child.id,
                source_program_id=source_b.id,
                role="top_k",
                order_index=0,
                weight=None,
                metadata={},
            ),
        ]

        summary = next(
            item for item in program_reads.get_summaries(session) if item["id"] == child.id
        )
        assert summary["archive_inspiration_ids"] == [source_a.id]
        assert summary["top_k_inspiration_ids"] == [source_b.id]
        assert list_sources_for_child(
            session,
            child.id,
            role="archive",
        ) == [source_a.id]
        assert list_sources_for_child(
            session,
            child.id,
            role="top_k",
        ) == [source_b.id]
        assert list_children_for_source(
            session,
            source_a.id,
            role="archive",
        ) == [child.id]
        assert list_children_for_source(
            session,
            source_b.id,
            role="top_k",
        ) == [child.id]
        assert count_usage_by_source(session, source_a.id) == 1
        assert count_usage_by_role(session, "archive") == 1
        assert count_usage_by_role(session, "top_k") == 1

    db.close()


def test_context_sampler_uses_repository_backed_archive_and_parent_selection(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    db = Database.open(db_path=str(db_path), num_islands=1)

    p0 = make_program(generation=0, score=1.0, correct=True, island_idx=0, timestamp=1.0)
    p1 = make_program(generation=1, score=2.0, correct=True, island_idx=0, timestamp=2.0)
    p2 = make_program(generation=2, score=5.0, correct=True, island_idx=0, timestamp=3.0)
    p3 = make_program(generation=3, score=4.0, correct=True, island_idx=0, timestamp=4.0)

    with db.session_scope() as session:
        for program in [p0, p1, p2, p3]:
            program_writes.add_program(session, program)

    sampled = ContextSampler(
        db,
        archive_policy=FitnessArchivePolicy(archive_size=3),
        parent_selection_strategy="winner_take_all",
        num_archive_inspirations=2,
        num_top_k_inspirations=1,
        enforce_island_separation=True,
    ).sample(target_generation=4)

    assert isinstance(sampled, SampledContext)
    assert sampled.parent.id == p2.id
    archive_ids = {program.id for program in sampled.archive_inspirations}
    topk_ids = {program.id for program in sampled.top_k_inspirations}
    assert p2.id not in archive_ids
    assert p2.id not in topk_ids
    assert len(sampled.archive_inspirations) <= 2
    assert len(sampled.top_k_inspirations) <= 1

    db.close()


def test_context_sampler_fix_mode_returns_incorrect_parent_with_ancestry(tmp_path):
    db_path = tmp_path / "programs.sqlite"
    db = Database.open(db_path=str(db_path), num_islands=1)

    root = make_program(generation=0, score=0.0, correct=False, island_idx=0, timestamp=1.0)
    child = make_program(
        generation=1,
        score=0.0,
        correct=False,
        island_idx=0,
        parent_id=root.id,
        timestamp=2.0,
    )
    with db.session_scope() as session:
        program_writes.add_program(session, root)
        program_writes.add_program(session, child)

    sampled = ContextSampler(
        db,
        archive_policy=FitnessArchivePolicy(archive_size=3),
        num_archive_inspirations=2,
        num_top_k_inspirations=1,
    ).sample(target_generation=2, with_fix_mode=True)

    assert sampled.needs_fix is True
    assert sampled.parent.id in {root.id, child.id}
    if sampled.parent.id == child.id:
        assert [program.id for program in sampled.archive_inspirations] == [root.id]

    db.close()
