import tempfile
from pathlib import Path

from shinka.controllers import (
    DatabaseController,
    RunStateController,
)
from shinka.controllers.metadata_controller import MetadataController
from shinka.database import Program


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
        controller = DatabaseController.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        db = controller.programs
        try:
            metadata = MetadataController(controller)
            metadata.set("custom_key", "custom_value")
            assert metadata.get("custom_key") == "custom_value"
        finally:
            db.close()


def test_run_state_controller_loads_and_persists_typed_run_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "run_state.db"
        controller = DatabaseController.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        db = controller.programs
        try:
            db.add(_program("prog-1", generation=0, island_idx=0))
            db.add(_program("prog-2", generation=1, island_idx=1))
            run_state = RunStateController(controller)
            run_state.set("best_program_id", "prog-1")
            run_state.set("beam_search_parent_id", "prog-2")
            run_state.set("best_score_generation", "7")
            run_state.set("best_score_ever", "12.5")
            snapshot = run_state.load_snapshot()

            assert snapshot.best_program_id == "prog-1"
            assert snapshot.beam_search_parent_id == "prog-2"
            assert snapshot.best_score_generation == 7
            assert snapshot.best_score_ever == 12.5
        finally:
            db.close()


def test_island_controller_reports_island_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "island_repo.db"
        root_controller = DatabaseController.open(
            db_path=str(db_path),
            num_islands=3,
            read_only=False,
        )
        db = root_controller.programs
        try:
            db.add(_program("p0", generation=0, island_idx=0))
            db.add(_program("p1", generation=1, island_idx=2))

            program_controller = root_controller.programs
            island_controller = root_controller.islands

            assert island_controller.get_program_island("p1") == 2
            initialized = island_controller.list_initialized_islands()
            assert [island.island_idx for island in initialized] == [0, 2]
            islands = island_controller.list_islands()
            assert [island.island_idx for island in islands] == [0, 1, 2]
            assert islands[1].initialized is False
            assert island_controller.get_island_populations() == {0: 1, 1: 0, 2: 1}
            assert island_controller.get_next_island_index() == 3
            assert program_controller.get_best_program_row()["id"] == "p1"
            assert program_controller.get_initial_program_row()["id"] == "p0"
        finally:
            db.close()


def test_database_controller_facade_exposes_metadata_inspirations_and_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "controller_facade.db"
        controller = DatabaseController.open(
            db_path=str(db_path),
            num_islands=2,
            read_only=False,
        )
        db = controller.programs
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
            db.add(parent)
            db.add(child)

            controller.metadata.set("custom_key", "custom_value")
            assert controller.metadata.get("custom_key") == "custom_value"

            assert controller.inspirations.list_sources_for_child("child") == [
                "parent",
                "parent",
            ]
            assert controller.inspirations.count_usage_by_role("archive") == 1

            controller.embeddings.update_features(
                program_id="child",
                embedding_pca_2d=[1.0, 2.0],
                embedding_pca_3d=[1.0, 2.0, 3.0],
                embedding_cluster_id=7,
            )
            all_embeddings = controller.embeddings.list_all()
            assert isinstance(all_embeddings, list)
        finally:
            db.close()
