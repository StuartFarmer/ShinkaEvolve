import asyncio
import tempfile
from pathlib import Path

from shinka.database import Program, ProgramRepository
from shinka.database.async_dbase import AsyncProgramDatabase


def _program(program_id: str) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=0,
    )


def test_program_database_init_without_openai_key(monkeypatch):
    """Repository-backed storage construction should not require API credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "no_key_init.db"
        db = ProgramRepository(str(db_path), num_islands=1, read_only=False)
        try:
            db.add(_program("p0"))
            repo = ProgramRepository(str(db_path), num_islands=1, read_only=True)
            assert repo.get("p0") is not None
            repo.close()
        finally:
            db.close()


def test_async_db_add_without_openai_key_when_embeddings_disabled(monkeypatch):
    """Async wrapper should preserve disabled embedding mode in worker DBs."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "no_key_async.db"
            last_iteration = 0
            beam_search_parent_id = None
            async_db = AsyncProgramDatabase(
                db_path=str(db_path),
                num_islands=1,
                embedding_model="",
                ensure_embedding_client=lambda: None,
                update_last_iteration=lambda value: max(last_iteration, value),
                update_beam_search_parent=lambda parent_id: parent_id,
            )
            try:
                await async_db.add_program_async(_program("async-p0"))
                repo = ProgramRepository(str(db_path), num_islands=1, read_only=True)
                assert repo.get("async-p0") is not None
                repo.close()
            finally:
                await async_db.close_async()

    asyncio.run(_run())
