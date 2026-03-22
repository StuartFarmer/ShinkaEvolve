"""Deprecated compatibility layer for parent selection.

Parent-selection logic now lives in `shinka.core.search_policies.ParentSelector`
and operates on repository-backed `Program` objects instead of raw SQLite rows.
This module remains only to avoid breaking older imports while the rest of the
codebase migrates.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Optional, Tuple

from .archive_policy import create_archive_policy
from shinka.controllers import DatabaseController, ProgramController
from .connector import DatabaseConnector
from shinka.core.search_policies import ParentSelector as RepositoryParentSelector

logger = logging.getLogger(__name__)


class CombinedParentSelector:
    """Deprecated shim that forwards parent sampling to the repository-backed policy layer."""

    def __init__(
        self,
        cursor,
        conn,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        beam_search_parent_id: Optional[str] = None,
        last_iteration: int = 0,
        update_metadata_func: Optional[Callable[[str, Optional[str]], None]] = None,
        get_best_program_func: Optional[Callable[[], Any]] = None,
    ):
        _ = (cursor, conn, get_program_func, best_program_id, beam_search_parent_id, last_iteration, update_metadata_func, get_best_program_func)
        self.db_path = getattr(config, "db_path", None)
        self.num_islands = getattr(config, "num_islands", 2)
        self.parent_selection_strategy = getattr(config, "parent_selection_strategy", "weighted")
        self.exploitation_alpha = getattr(config, "exploitation_alpha", 1.0)
        self.parent_selection_lambda = getattr(config, "parent_selection_lambda", 10.0)
        self.num_beams = getattr(config, "num_beams", 5)
        self.archive_selection_strategy = getattr(config, "archive_selection_strategy", "fitness")
        self.archive_size = getattr(config, "archive_size", 40)
        self.archive_criteria = getattr(config, "archive_criteria", {"combined_score": 1.0})
        self.selector = RepositoryParentSelector(
            parent_selection_strategy=self.parent_selection_strategy,
            exploitation_alpha=self.exploitation_alpha,
            parent_selection_lambda=self.parent_selection_lambda,
            num_beams=self.num_beams,
        )
        self.archive_policy = create_archive_policy(
            archive_selection_strategy=self.archive_selection_strategy,
            archive_size=self.archive_size,
            archive_criteria=self.archive_criteria,
        )
        logger.warning(
            "CombinedParentSelector is deprecated. Use shinka.core.search_policies.ParentSelector."
        )

    def _repository(self) -> ProgramController:
        if not self.db_path:
            raise RuntimeError(
                "Legacy CombinedParentSelector requires config.db_path for repository-backed sampling."
            )
        warnings.warn(
            "CombinedParentSelector is deprecated; use DatabaseController().programs + "
            "shinka.core.search_policies.ParentSelector directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DatabaseController(
            DatabaseConnector.open(
                db_path=self.db_path,
                num_islands=self.num_islands,
                read_only=True,
            )
        ).programs

    def has_correct_programs(self, island_idx: Optional[int] = None) -> bool:
        repository = self._repository()
        try:
            return self.selector.has_correct_programs(repository, island_idx=island_idx)
        finally:
            repository.close()

    def get_incorrect_program_for_fix(
        self, island_idx: Optional[int] = None
    ) -> Optional[Any]:
        repository = self._repository()
        try:
            return self.selector.get_incorrect_program_for_fix(
                repository, island_idx=island_idx
            )
        finally:
            repository.close()

    def sample_parent_with_fix_mode(
        self, island_idx: Optional[int] = None
    ) -> Tuple[Any, bool]:
        repository = self._repository()
        try:
            archive = self.archive_policy.compute(repository.list_correct())
            return self.selector.select_with_fix_mode(
                repository,
                archive,
                island_idx=island_idx,
            )
        finally:
            repository.close()

    def sample_parent(self, island_idx: Optional[int] = None) -> Any:
        repository = self._repository()
        try:
            archive = self.archive_policy.compute(repository.list_correct())
            return self.selector.select(
                repository,
                archive,
                island_idx=island_idx,
            )
        finally:
            repository.close()
