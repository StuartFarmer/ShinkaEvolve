"""Deprecated compatibility layer for inspiration selection.

Inspiration selection now lives in `shinka.core.search_policies.InspirationSelector`
and works over repository-returned `Program` objects plus computed archive state.
This module remains only for older imports while the migration completes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Literal, Optional

from .archive_policy import create_archive_policy
from .repository import ProgramRepository
from shinka.core.search_policies import InspirationSelector as RepositoryInspirationSelector

logger = logging.getLogger(__name__)


class CombinedContextSelector:
    """Deprecated shim for repository-backed inspiration selection."""

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
        self.config = config
        self.selector = RepositoryInspirationSelector(config)
        self.archive_policy = create_archive_policy(config)
        logger.warning(
            "CombinedContextSelector is deprecated. Use shinka.core.search_policies.InspirationSelector."
        )

    def _repository(self) -> ProgramRepository:
        if not getattr(self.config, "db_path", None):
            raise RuntimeError(
                "Legacy CombinedContextSelector requires config.db_path for repository-backed sampling."
            )
        return ProgramRepository.from_config(self.config, read_only=True)

    def sample_context(
        self, parent: Any, num_archive: int, num_topk: int
    ) -> tuple[List[Any], List[Any]]:
        repository = self._repository()
        try:
            archive = self.archive_policy.compute(repository.list_correct())
            archive_inspirations = self.selector.select_archive(
                repository,
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
            repository.close()


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
