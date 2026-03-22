"""Deprecated compatibility layer for inspiration selection.

Inspiration selection now lives in `shinka.core.search_policies.InspirationSelector`
and works over controller-returned `Program` objects plus computed archive state.
This module remains only for older imports while the migration completes.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, List, Literal, Optional

from .archive_policy import create_archive_policy
from shinka.controllers import DatabaseController, ProgramController
from .connector import DatabaseConnector
from shinka.core.search_policies import InspirationSelector as RepositoryInspirationSelector

logger = logging.getLogger(__name__)


class CombinedContextSelector:
    """Deprecated shim for controller-backed inspiration selection."""

    def __init__(
        self,
        cursor,
        conn,
        config: Any,
        get_program_func: Callable[[str], Any],
        best_program_id: Optional[str] = None,
        get_island_idx_func: Optional[Callable[[str], Optional[int]]] = None,
        program_from_row_func: Optional[Callable[[Any], Any]] = None,
    ):
        _ = (cursor, conn, get_program_func, best_program_id, get_island_idx_func, program_from_row_func)
        self.db_path = getattr(config, "db_path", None)
        self.num_islands = getattr(config, "num_islands", 2)
        self.enforce_island_separation = getattr(config, "enforce_island_separation", False)
        self.elite_selection_ratio = getattr(config, "elite_selection_ratio", 0.3)
        self.archive_selection_strategy = getattr(config, "archive_selection_strategy", "fitness")
        self.archive_size = getattr(config, "archive_size", 40)
        self.archive_criteria = getattr(config, "archive_criteria", {"combined_score": 1.0})
        self.selector = RepositoryInspirationSelector(
            enforce_island_separation=self.enforce_island_separation,
            elite_selection_ratio=self.elite_selection_ratio,
        )
        self.archive_policy = create_archive_policy(
            archive_selection_strategy=self.archive_selection_strategy,
            archive_size=self.archive_size,
            archive_criteria=self.archive_criteria,
        )
        logger.warning(
            "CombinedContextSelector is deprecated. Use shinka.core.search_policies.InspirationSelector."
        )

    def _programs(self) -> ProgramController:
        if not self.db_path:
            raise RuntimeError(
                "Legacy CombinedContextSelector requires config.db_path for controller-backed sampling."
            )
        warnings.warn(
            "CombinedContextSelector is deprecated; use DatabaseController().programs + "
            "shinka.core.search_policies.InspirationSelector directly.",
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

    def sample_context(
        self, parent: Any, num_archive: int, num_topk: int
    ) -> tuple[List[Any], List[Any]]:
        programs = self._programs()
        try:
            archive = self.archive_policy.compute(programs.list_correct())
            archive_inspirations = self.selector.select_archive(
                programs,
                parent,
                archive,
                n=num_archive,
            )
            top_k_inspirations = self.selector.select_top_k(
                parent,
                archive,
                excluded_programs=archive_inspirations,
                k=num_topk,
            )
            return archive_inspirations, top_k_inspirations
        finally:
            programs.close()


class InspirationContextBuilder:
    """Compatibility helper for prompt presentation order."""

    def __init__(
        self,
        sort_order: Literal["ascending", "chronological", "none"] = "ascending",
    ):
        self.sort_order = sort_order

    def build(self, archive_inspirations: List[Any], top_k_inspirations: List[Any]) -> List[Any]:
        inspirations = list(archive_inspirations) + list(top_k_inspirations)
        if self.sort_order == "ascending":
            return sorted(
                inspirations,
                key=lambda program: (
                    float(program.combined_score or 0.0),
                    float(program.timestamp),
                ),
            )
        if self.sort_order == "chronological":
            return sorted(
                inspirations,
                key=lambda program: (int(program.generation), float(program.timestamp)),
            )
        return inspirations
