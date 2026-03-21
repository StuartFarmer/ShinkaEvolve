"""
Context sampling for proposal generation.

This module owns the first step in the proposal pipeline:

`sample context -> build prompt -> ...`

It composes smaller policy/services:
- `ProgramRepository` for persisted state
- `ArchivePolicy` for computed archive membership
- `ParentSelector` for lineage choice
- `InspirationSelector` for prompt conditioning

The old `ProgramDatabase.sample*` path is intentionally bypassed here.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from shinka.core.search_policies import InspirationSelector, ParentSelector
from shinka.database.dbase import Program, DatabaseConfig
from shinka.database.repository import ProgramRepository
from shinka.database.archive_policy import ArchivePolicy, create_archive_policy


@dataclass(frozen=True)
class SampledContext:
    """Typed input context for one proposal attempt."""

    parent: Program
    archive_inspirations: List[Program] = field(default_factory=list)
    top_k_inspirations: List[Program] = field(default_factory=list)
    needs_fix: bool = False
    sampled_island: Optional[int] = None
    target_generation: Optional[int] = None
    novelty_attempt: Optional[int] = None
    max_novelty_attempts: Optional[int] = None
    resample_attempt: Optional[int] = None
    max_resample_attempts: Optional[int] = None


class ContextSampler:
    """Repository-backed context sampler for one proposal attempt."""

    def __init__(
        self,
        repository: ProgramRepository,
        *,
        archive_policy: Optional[ArchivePolicy] = None,
        parent_selector: Optional[ParentSelector] = None,
        inspiration_selector: Optional[InspirationSelector] = None,
    ):
        self.repository = repository
        self.config = repository.config
        self.archive_policy = archive_policy or create_archive_policy(repository.config)
        self.parent_selector = parent_selector or ParentSelector(repository.config)
        self.inspiration_selector = inspiration_selector or InspirationSelector(
            repository.config
        )

    def sample(
        self,
        *,
        target_generation: Optional[int] = None,
        novelty_attempt: Optional[int] = None,
        max_novelty_attempts: Optional[int] = None,
        resample_attempt: Optional[int] = None,
        max_resample_attempts: Optional[int] = None,
        with_fix_mode: bool = True,
    ) -> SampledContext:
        if not self._are_all_islands_initialized():
            return self._sample_during_island_bootstrap(
                target_generation=target_generation,
                novelty_attempt=novelty_attempt,
                max_novelty_attempts=max_novelty_attempts,
                resample_attempt=resample_attempt,
                max_resample_attempts=max_resample_attempts,
                with_fix_mode=with_fix_mode,
            )

        initialized_islands = self.repository.list_initialized_islands()
        sampled_island = self._sample_island(initialized_islands)

        archive_programs = self._compute_archive()
        if with_fix_mode:
            parent, needs_fix = self.parent_selector.select_with_fix_mode(
                self.repository,
                archive_programs,
                island_idx=sampled_island,
            )
        else:
            parent = self.parent_selector.select(
                self.repository,
                archive_programs,
                island_idx=sampled_island,
            )
            needs_fix = False

        if needs_fix:
            num_ancestors = int(getattr(self.config, "num_archive_inspirations", 0)) + int(
                getattr(self.config, "num_top_k_inspirations", 0)
            )
            ancestor_inspirations = self.repository.get_ancestry(
                parent.id,
                max_ancestors=num_ancestors,
            )
            return SampledContext(
                parent=parent,
                archive_inspirations=ancestor_inspirations,
                top_k_inspirations=[],
                needs_fix=True,
                sampled_island=sampled_island,
                target_generation=target_generation,
                novelty_attempt=novelty_attempt,
                max_novelty_attempts=max_novelty_attempts,
                resample_attempt=resample_attempt,
                max_resample_attempts=max_resample_attempts,
            )

        num_archive = int(getattr(self.config, "num_archive_inspirations", 0))
        num_topk = int(getattr(self.config, "num_top_k_inspirations", 0))
        archive_inspirations = self.inspiration_selector.select_archive(
            self.repository,
            parent,
            archive_programs,
            n=num_archive,
        )
        top_k_inspirations = self.inspiration_selector.select_top_k(
            parent,
            archive_programs,
            excluded_programs=archive_inspirations,
            k=num_topk,
        )

        return SampledContext(
            parent=parent,
            archive_inspirations=archive_inspirations,
            top_k_inspirations=top_k_inspirations,
            needs_fix=False,
            sampled_island=sampled_island,
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
            max_novelty_attempts=max_novelty_attempts,
            resample_attempt=resample_attempt,
            max_resample_attempts=max_resample_attempts,
        )

    def _compute_archive(self) -> List[Program]:
        return self.archive_policy.compute(self.repository.list_correct())

    def _are_all_islands_initialized(self) -> bool:
        initialized = self.repository.list_initialized_islands()
        if not initialized:
            return False
        num_islands = int(getattr(self.config, "num_islands", 1))
        if num_islands <= 1:
            return True
        return len(initialized) >= num_islands

    def _sample_during_island_bootstrap(
        self,
        *,
        target_generation: Optional[int],
        novelty_attempt: Optional[int],
        max_novelty_attempts: Optional[int],
        resample_attempt: Optional[int],
        max_resample_attempts: Optional[int],
        with_fix_mode: bool,
    ) -> SampledContext:
        correct_programs = self.repository.list_correct()

        if correct_programs:
            parent = self.repository.get_earliest()
            if parent is None:
                raise RuntimeError("No programs found in repository")
            needs_fix = with_fix_mode and not parent.correct
            archive_inspirations: List[Program] = []
            if needs_fix:
                num_ancestors = int(getattr(self.config, "num_archive_inspirations", 0)) + int(
                    getattr(self.config, "num_top_k_inspirations", 0)
                )
                archive_inspirations = self.repository.get_ancestry(
                    parent.id,
                    max_ancestors=num_ancestors,
                )
            return SampledContext(
                parent=parent,
                archive_inspirations=archive_inspirations,
                top_k_inspirations=[],
                needs_fix=needs_fix,
                sampled_island=parent.island_idx,
                target_generation=target_generation,
                novelty_attempt=novelty_attempt,
                max_novelty_attempts=max_novelty_attempts,
                resample_attempt=resample_attempt,
                max_resample_attempts=max_resample_attempts,
            )

        incorrect_programs = self.repository.list_incorrect()
        if incorrect_programs:
            parent = random.choice(incorrect_programs)
            num_ancestors = int(getattr(self.config, "num_archive_inspirations", 0)) + int(
                getattr(self.config, "num_top_k_inspirations", 0)
            )
            archive_inspirations = (
                self.repository.get_ancestry(parent.id, max_ancestors=num_ancestors)
                if with_fix_mode
                else []
            )
            return SampledContext(
                parent=parent,
                archive_inspirations=archive_inspirations,
                top_k_inspirations=[],
                needs_fix=with_fix_mode,
                sampled_island=parent.island_idx,
                target_generation=target_generation,
                novelty_attempt=novelty_attempt,
                max_novelty_attempts=max_novelty_attempts,
                resample_attempt=resample_attempt,
                max_resample_attempts=max_resample_attempts,
            )

        parent = self.repository.get_earliest()
        if parent is None:
            raise RuntimeError("No programs found in repository")
        return SampledContext(
            parent=parent,
            archive_inspirations=[],
            top_k_inspirations=[],
            needs_fix=with_fix_mode and not parent.correct,
            sampled_island=parent.island_idx,
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
            max_novelty_attempts=max_novelty_attempts,
            resample_attempt=resample_attempt,
            max_resample_attempts=max_resample_attempts,
        )

    def _sample_island(self, initialized_islands: List[int]) -> int:
        if not initialized_islands:
            raise ValueError("No initialized islands available for sampling.")

        strategy = getattr(self.config, "island_selection_strategy", "uniform")
        if strategy == "uniform":
            return random.choice(initialized_islands)

        if strategy == "equal":
            counts = self.repository.get_island_program_counts(initialized_islands)
            min_count = min(counts.values())
            candidates = [idx for idx, count in counts.items() if count == min_count]
            return random.choice(candidates)

        if strategy == "proportional":
            fitness = self.repository.get_island_best_scores(initialized_islands)
            values = np.array([fitness.get(idx, 0.0) for idx in initialized_islands], dtype=float)
            exp_values = np.exp(values)
            probs = exp_values / np.sum(exp_values) if float(np.sum(exp_values)) > 0 else np.ones(len(values)) / len(values)
            return initialized_islands[int(np.random.choice(len(initialized_islands), p=probs))]

        if strategy == "weighted":
            counts = self.repository.get_island_program_counts(initialized_islands)
            fitness = self.repository.get_island_best_scores(initialized_islands)
            weights = []
            for island_idx in initialized_islands:
                count = counts.get(island_idx, 1) or 1
                best_score = max(fitness.get(island_idx, 0.0), 0.0)
                weights.append((best_score + 1e-6) / count)
            weights_arr = np.array(weights, dtype=float)
            probs = weights_arr / np.sum(weights_arr) if float(np.sum(weights_arr)) > 0 else np.ones(len(weights_arr)) / len(weights_arr)
            return initialized_islands[int(np.random.choice(len(initialized_islands), p=probs))]

        raise ValueError(f"Unknown island selection strategy: {strategy}")


class AsyncContextSampler:
    """
    Async sibling for the async runner.

    Each call opens a fresh read-only repository, which keeps sampling isolated
    from concurrent writer state and avoids shared-cursor coupling.
    """

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self._lock = asyncio.Lock()

    async def sample(
        self,
        *,
        target_generation: Optional[int] = None,
        novelty_attempt: Optional[int] = None,
        max_novelty_attempts: Optional[int] = None,
        resample_attempt: Optional[int] = None,
        max_resample_attempts: Optional[int] = None,
        with_fix_mode: bool = True,
    ) -> SampledContext:
        async with self._lock:
            repository = ProgramRepository.from_config(self.db_config, read_only=True)
            try:
                sampler = ContextSampler(repository)
                return sampler.sample(
                    target_generation=target_generation,
                    novelty_attempt=novelty_attempt,
                    max_novelty_attempts=max_novelty_attempts,
                    resample_attempt=resample_attempt,
                    max_resample_attempts=max_resample_attempts,
                    with_fix_mode=with_fix_mode,
                )
            finally:
                repository.close()
