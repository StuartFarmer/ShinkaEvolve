import tempfile
from pathlib import Path

from shinka.database import (
    Database,
    Program,
    embedding_ops,
    inspiration_ops,
    island_ops,
    metadata_ops,
    program_reads,
    program_writes,
    run_state_ops,
)


def _program(program_id: str, *, generation: int = 0, island_idx: int = 0) -> Program:
    return Program(
        id=program_id,
        code="def run():\n    return 1\n",
        correct=True,
        combined_score=float(generation + 1),
        generation=generation,
        island_idx=island_idx,
    )


def test_metadata_controller_loads_and_persists_generic_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metadata_repo.db"
        db = Database.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        try:
            with db.session_scope() as session:
                metadata_ops.set(session, "custom_key", "custom_value")
            with db.session() as session:
                assert metadata_ops.get(session, "custom_key") == "custom_value"
        finally:
            db.close()


def test_run_state_controller_loads_and_persists_typed_run_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "run_state.db"
        db = Database.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        try:
            with db.session_scope() as session:
                program_writes.add_program(session, _program("prog-1", generation=0, island_idx=0))
                program_writes.add_program(session, _program("prog-2", generation=1, island_idx=1))
                run_state_ops.set(session, "best_program_id", "prog-1")
                run_state_ops.set(session, "beam_search_parent_id", "prog-2")
                run_state_ops.set(session, "best_score_generation", "7")
                run_state_ops.set(session, "best_score_ever", "12.5")
            with db.session() as session:
                snapshot = run_state_ops.load_snapshot(session, read_only=True)

            assert snapshot.best_program_id == "prog-1"
            assert snapshot.beam_search_parent_id == "prog-2"
            assert snapshot.best_score_generation == 7
            assert snapshot.best_score_ever == 12.5
        finally:
            db.close()


def test_island_controller_reports_island_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "island_repo.db"
        db = Database.open(
            db_path=str(db_path),
            num_islands=3,
            read_only=False,
        )
        try:
            with db.session_scope() as session:
                program_writes.add_program(session, _program("p0", generation=0, island_idx=0))
                program_writes.add_program(session, _program("p1", generation=1, island_idx=2))
            with db.session() as session:
                assert island_ops.get_program_island(session, "p1") == 2
                initialized = island_ops.list_initialized_islands(session, num_islands=3)
                islands = island_ops.list_islands(session, num_islands=3)
                assert island_ops.get_island_populations(session, num_islands=3) == {
                    0: 1,
                    1: 0,
                    2: 1,
                }
                assert island_ops.get_next_island_index(session, num_islands=3) == 3
                assert program_reads.get_best_program_row(session)["id"] == "p1"
                assert program_reads.get_initial_program_row(session)["id"] == "p0"

            assert [island.island_idx for island in initialized] == [0, 2]
            assert [island.island_idx for island in islands] == [0, 1, 2]
            assert islands[1].initialized is False
        finally:
            db.close()

def test_database_ops_cover_metadata_inspirations_and_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "database_ops.db"
        db = Database.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        try:
            parent = _program("parent", generation=0, island_idx=0)
            child = Program(
                id="child",
                code="def run():\n    return 2\n",
                correct=True,
                combined_score=2.0,
                generation=1,
                island_idx=0,
                parent_id="parent",
                archive_inspiration_ids=["parent"],
                top_k_inspiration_ids=["parent"],
            )
            with db.session_scope() as session:
                program_writes.add_program(session, parent)
                program_writes.add_program(session, child)
                metadata_ops.set(session, "custom_key", "custom_value")
                embedding_ops.update_features(
                    session,
                    program_id="child",
                    embedding_pca_2d=[1.0, 2.0],
                    embedding_pca_3d=[1.0, 2.0, 3.0],
                    embedding_cluster_id=7,
                )

            with db.session() as session:
                assert metadata_ops.get(session, "custom_key") == "custom_value"
                assert inspiration_ops.list_sources_for_child(session, "child") == [
                    "parent",
                    "parent",
                ]
                assert inspiration_ops.count_usage_by_role(session, "archive") == 1
                all_embeddings = embedding_ops.list_all(session)
                assert isinstance(all_embeddings, list)
        finally:
            db.close()
