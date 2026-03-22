from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.controllers import DatabaseController
from shinka.database import Program
from shinka.database.connector import DatabaseConnector


def test_explicit_island_assignment_is_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "explicit_island.db"
        connector = DatabaseConnector.open(db_path=str(db_path), num_islands=3, read_only=False)
        db = DatabaseController(connector).programs

        seeded_program = Program(
            id="seed_program",
            code="def run():\n    return 0\n",
            generation=0,
            correct=True,
            combined_score=1.0,
            island_idx=2,
            metadata={"family_id": "family_two"},
        )

        try:
            db.add(seeded_program)
            repo = DatabaseController(
                DatabaseConnector.open(db_path=str(db_path), num_islands=3, read_only=True)
            ).programs
            stored_program = repo.get("seed_program")
            assert stored_program is not None
            assert stored_program.island_idx == 2
            assert repo.list_by_generation(0) == [stored_program]
        finally:
            repo.close()

        db.close()


def test_island_family_context_is_composed_into_system_prompt():
    runner = ShinkaEvolveRunner.__new__(ShinkaEvolveRunner)
    runner.evo_config = SimpleNamespace(task_sys_msg="Base task prompt")

    parent_program = Program(
        id="parent",
        code="def run():\n    return 0\n",
        metadata={
            "family_id": "mean_variance",
            "family_name": "Mean Variance Control",
            "family_context": "Focus on certainty-equivalent position sizing.",
        },
    )

    prompt = runner._compose_island_system_prompt("Global evolved prompt", parent_program)

    assert prompt is not None
    assert "Global evolved prompt" in prompt
    assert "# Island Family Context" in prompt
    assert "Mean Variance Control" in prompt
    assert "certainty-equivalent position sizing" in prompt
