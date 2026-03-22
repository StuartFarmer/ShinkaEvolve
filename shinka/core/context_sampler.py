"""
Context sampling for proposal generation.

This module owns the first step in the proposal pipeline:

`sample context -> build prompt -> ...`

It composes smaller policy/services:
- `ProgramController` for persisted state
- `ArchivePolicy` for computed archive membership
- `ParentSelector` for lineage choice
- `InspirationSelector` for prompt conditioning

The old `ProgramDatabase.sample*` path is intentionally bypassed here.
"""

from __future__ import annotations

import asyncio
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from shinka.core.search_policies import InspirationSelector, ParentSelector
from shinka.controllers import DatabaseController, ProgramController
from shinka.controllers.types import Island
from shinka.database.program import Program
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
    """Controller-backed context sampler for one proposal attempt."""

    def __init__(
        self,
        programs: ProgramController,
        *,
        archive_policy: Optional[ArchivePolicy] = None,
        parent_selector: Optional[ParentSelector] = None,
        inspiration_selector: Optional[InspirationSelector] = None,
        num_islands: Optional[int] = None,
        island_selection_strategy: str = "uniform",
        num_archive_inspirations: int = 1,
        num_top_k_inspirations: int = 1,
        parent_selection_strategy: str = "weighted",
        exploitation_alpha: float = 1.0,
        parent_selection_lambda: float = 10.0,
        num_beams: int = 5,
        enforce_island_separation: bool = True,
        elite_selection_ratio: float = 0.3,
    ):
        self.programs = programs
        self.num_islands = (
            programs.num_islands if num_islands is None else num_islands
        )

        self.island_selection_strategy = island_selection_strategy
        self.num_archive_inspirations = num_archive_inspirations
        self.num_top_k_inspirations = num_top_k_inspirations
        self.archive_policy = archive_policy or create_archive_policy()
        self.parent_selector = parent_selector or ParentSelector(
            parent_selection_strategy=parent_selection_strategy,
            exploitation_alpha=exploitation_alpha,
            parent_selection_lambda=parent_selection_lambda,
            num_beams=num_beams,
        )
        self.inspiration_selector = inspiration_selector or InspirationSelector(
            enforce_island_separation=enforce_island_separation,
            elite_selection_ratio=elite_selection_ratio,
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

        initialized_islands = self.programs.list_initialized_islands()
        sampled_island = self._sample_island(initialized_islands)

        archive_programs = self._compute_archive()
        if with_fix_mode:
            parent, needs_fix = self.parent_selector.select_with_fix_mode(
                self.programs,
                archive_programs,
                island_idx=sampled_island,
            )
        else:
            parent = self.parent_selector.select(
                self.programs,
                archive_programs,
                island_idx=sampled_island,
            )
            needs_fix = False

        if needs_fix:
            num_ancestors = self.num_archive_inspirations + self.num_top_k_inspirations
            ancestor_inspirations = self.programs.get_ancestry(
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

        num_archive = self.num_archive_inspirations
        num_topk = self.num_top_k_inspirations
        archive_inspirations = self.inspiration_selector.select_archive(
            self.programs,
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
        return self.archive_policy.compute(self.programs.list_correct())

    def _are_all_islands_initialized(self) -> bool:
        initialized = self.programs.list_initialized_island_ids()
        if not initialized:
            return False
        num_islands = int(self.num_islands)
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
        correct_programs = self.programs.list_correct()

        if correct_programs:
            parent = self.programs.get_earliest()
            if parent is None:
                raise RuntimeError("No programs found in program controller")
            needs_fix = with_fix_mode and not parent.correct
            archive_inspirations: List[Program] = []
            if needs_fix:
                num_ancestors = self.num_archive_inspirations + self.num_top_k_inspirations
                archive_inspirations = self.programs.get_ancestry(
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

        incorrect_programs = self.programs.list_incorrect()
        if incorrect_programs:
            parent = random.choice(incorrect_programs)
            num_ancestors = self.num_archive_inspirations + self.num_top_k_inspirations
            archive_inspirations = (
                self.programs.get_ancestry(parent.id, max_ancestors=num_ancestors)
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

        parent = self.programs.get_earliest()
        if parent is None:
            raise RuntimeError("No programs found in program controller")
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

    def _sample_island(self, initialized_islands: List[Island]) -> int:
        if not initialized_islands:
            raise ValueError("No initialized islands available for sampling.")

        strategy = self.island_selection_strategy
        if strategy == "uniform":
            return random.choice(initialized_islands).island_idx

        if strategy == "equal":
            min_count = min(island.correct_programs for island in initialized_islands)
            candidates = [
                island.island_idx
                for island in initialized_islands
                if island.correct_programs == min_count
            ]
            return random.choice(candidates)

        if strategy == "proportional":
            values = np.array(
                [island.best_score for island in initialized_islands],
                dtype=float,
            )
            exp_values = np.exp(values)
            probs = (
                exp_values / np.sum(exp_values)
                if float(np.sum(exp_values)) > 0
                else np.ones(len(values)) / len(values)
            )
            return initialized_islands[
                int(np.random.choice(len(initialized_islands), p=probs))
            ].island_idx

        if strategy == "weighted":
            weights = []
            for island in initialized_islands:
                count = island.correct_programs or 1
                best_score = max(island.best_score, 0.0)
                weights.append((best_score + 1e-6) / count)
            weights_arr = np.array(weights, dtype=float)
            probs = (
                weights_arr / np.sum(weights_arr)
                if float(np.sum(weights_arr)) > 0
                else np.ones(len(weights_arr)) / len(weights_arr)
            )
            return initialized_islands[
                int(np.random.choice(len(initialized_islands), p=probs))
            ].island_idx

        raise ValueError(f"Unknown island selection strategy: {strategy}")


class AsyncContextSampler:
    """
    Async sibling for the async runner.

    Each call opens a fresh read-only controller, which keeps sampling isolated
    from concurrent writer state and avoids shared-cursor coupling.
    """

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        num_islands: int = 2,
        island_selection_strategy: str = "uniform",
        num_archive_inspirations: int = 1,
        num_top_k_inspirations: int = 1,
        parent_selection_strategy: str = "weighted",
        exploitation_alpha: float = 1.0,
        parent_selection_lambda: float = 10.0,
        num_beams: int = 5,
        enforce_island_separation: bool = True,
        elite_selection_ratio: float = 0.3,
    ):
        self.db_path = db_path
        self.num_islands = num_islands
        self.island_selection_strategy = island_selection_strategy
        self.num_archive_inspirations = num_archive_inspirations
        self.num_top_k_inspirations = num_top_k_inspirations
        self.parent_selection_strategy = parent_selection_strategy
        self.exploitation_alpha = exploitation_alpha
        self.parent_selection_lambda = parent_selection_lambda
        self.num_beams = num_beams
        self.enforce_island_separation = enforce_island_separation
        self.elite_selection_ratio = elite_selection_ratio
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
            programs = DatabaseController.open(
                db_path=self.db_path,
                num_islands=self.num_islands,
                read_only=True,
            ).programs
            try:
                sampler = ContextSampler(
                    programs,
                    num_islands=self.num_islands,
                    island_selection_strategy=self.island_selection_strategy,
                    num_archive_inspirations=self.num_archive_inspirations,
                    num_top_k_inspirations=self.num_top_k_inspirations,
                    parent_selection_strategy=self.parent_selection_strategy,
                    exploitation_alpha=self.exploitation_alpha,
                    parent_selection_lambda=self.parent_selection_lambda,
                    num_beams=self.num_beams,
                    enforce_island_separation=self.enforce_island_separation,
                    elite_selection_ratio=self.elite_selection_ratio,
                )
                return sampler.sample(
                    target_generation=target_generation,
                    novelty_attempt=novelty_attempt,
                    max_novelty_attempts=max_novelty_attempts,
                    resample_attempt=resample_attempt,
                    max_resample_attempts=max_resample_attempts,
                    with_fix_mode=with_fix_mode,
                )
            finally:
                programs.close()
