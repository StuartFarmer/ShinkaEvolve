import tempfile
from pathlib import Path

from shinka.database import (
    DatabaseConfig,
    IslandRepository,
    MetadataRepository,
    Program,
    ProgramDatabase,
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


def test_metadata_repository_loads_and_persists_run_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "metadata_repo.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=2),
            embedding_model="",
            read_only=False,
        )
        try:
            repo = MetadataRepository(conn=db.conn, cursor=db.cursor, read_only=False)
            repo.set("best_program_id", "prog-1")
            repo.set("beam_search_parent_id", "prog-2")
            repo.set("best_score_generation", "7")
            repo.set("best_score_ever", "12.5")
            snapshot = repo.load_snapshot()

            assert snapshot.best_program_id == "prog-1"
            assert snapshot.beam_search_parent_id == "prog-2"
            assert snapshot.best_score_generation == 7
            assert snapshot.best_score_ever == 12.5
        finally:
            db.close()


def test_island_repository_reports_island_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "island_repo.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=3),
            embedding_model="",
            read_only=False,
        )
        try:
            db.add(_program("p0", generation=0, island_idx=0))
            db.add(_program("p1", generation=1, island_idx=2))

            repo = IslandRepository(
                conn=db.conn,
                cursor=db.cursor,
                num_islands=db.config.num_islands,
            )

            assert repo.get_program_island("p1") == 2
            initialized = repo.list_initialized_islands()
            assert [island.island_idx for island in initialized] == [0, 2]
            islands = repo.list_islands()
            assert [island.island_idx for island in islands] == [0, 1, 2]
            assert islands[1].initialized is False
            assert repo.get_island_populations() == {0: 1, 1: 0, 2: 1}
            assert repo.get_next_island_index() == 3
            assert repo.get_best_program_row()["id"] == "p1"
            assert repo.get_initial_program_row()["id"] == "p0"
        finally:
            db.close()
