"""Island sampling strategies for parent selection."""

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import numpy as np
from shinka.controllers.types import Island

if TYPE_CHECKING:
    from shinka.controllers.program_controller import ProgramController

logger = logging.getLogger(__name__)


class IslandSampler(ABC):
    """Abstract base class for island sampling strategies."""

    def __init__(
        self,
        programs: "ProgramController",
    ):
        self.programs = programs

    @abstractmethod
    def sample_island(self, initialized_islands: List[Island]) -> int:
        """Sample an island index from the list of initialized islands.

        Args:
            initialized_islands: List of island indices that have correct programs

        Returns:
            Selected island index
        """
        pass

    def _normalize_islands(self, initialized_islands: List[Island | int]) -> List[Island]:
        island_map = {
            island.island_idx: island for island in self.programs.list_islands()
        }
        normalized: List[Island] = []
        for island in initialized_islands:
            if isinstance(island, Island):
                normalized.append(island)
            else:
                normalized.append(island_map[int(island)])
        return normalized


class UniformIslandSampler(IslandSampler):
    """Uniformly sample from initialized islands (default behavior)."""

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        """Uniformly sample an island."""
        initialized_islands = self._normalize_islands(initialized_islands)
        return random.choice(initialized_islands).island_idx


class EqualIslandSampler(IslandSampler):
    """Sample the island with the fewest programs.

    If multiple islands have the same minimum count, sample uniformly among them.
    """

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        """Sample island with fewest programs."""
        initialized_islands = self._normalize_islands(initialized_islands)
        min_count = min(island.correct_programs for island in initialized_islands)
        islands_with_min = [
            island.island_idx
            for island in initialized_islands
            if island.correct_programs == min_count
        ]

        sampled = random.choice(islands_with_min)
        logger.debug(
            f"EqualIslandSampler: Island counts = {[island.correct_programs for island in initialized_islands]}, "
            f"min_count = {min_count}, sampled = {sampled}"
        )
        return sampled


class ProportionalIslandSampler(IslandSampler):
    """Sample islands proportional to their best fitness using Boltzmann distribution.

    Uses a medium temperature for the Boltzmann distribution.
    """

    def __init__(
        self,
        programs: "ProgramController",
        temperature: float = 1.0,
    ):
        super().__init__(programs)
        self.temperature = temperature

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        """Sample island proportional to best fitness."""
        initialized_islands = self._normalize_islands(initialized_islands)
        fitness_values = np.array(
            [island.best_score for island in initialized_islands]
        )

        # Apply Boltzmann distribution: exp(fitness / temperature)
        exp_values = np.exp(fitness_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx].island_idx

        logger.debug(
            f"ProportionalIslandSampler: fitness = {[island.best_score for island in initialized_islands]}, "
            f"probabilities = {probabilities}, sampled = {sampled_island}"
        )
        return sampled_island


class WeightedIslandSampler(IslandSampler):
    """Sample islands considering both program count and fitness.

    More programs -> lower probability
    Higher fitness -> higher probability
    """

    def __init__(
        self,
        programs: "ProgramController",
        fitness_weight: float = 1.0,
        count_weight: float = 1.0,
    ):
        super().__init__(programs)
        self.fitness_weight = fitness_weight
        self.count_weight = count_weight

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        """Sample island using weighted combination of fitness and inverse count."""
        initialized_islands = self._normalize_islands(initialized_islands)
        # Calculate weights for each island
        weights = []
        for island in initialized_islands:
            count = island.correct_programs or 1
            fitness = island.best_score

            # Weight = fitness^fitness_weight / count^count_weight
            # More fitness -> higher weight, more programs -> lower weight
            weight = (fitness**self.fitness_weight) / (count**self.count_weight)
            weights.append(weight)

        # Normalize to probabilities
        weights = np.array(weights)
        if np.sum(weights) == 0:
            # Fallback to uniform if all weights are zero
            probabilities = np.ones(len(weights)) / len(weights)
        else:
            probabilities = weights / np.sum(weights)

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx].island_idx

        logger.debug(
            f"WeightedIslandSampler: counts = {[island.correct_programs for island in initialized_islands]}, "
            f"fitness = {[island.best_score for island in initialized_islands]}, "
            f"weights = {weights}, probabilities = {probabilities}, "
            f"sampled = {sampled_island}"
        )
        return sampled_island


def create_island_sampler(
    programs: "ProgramController",
    strategy: str = "uniform",
) -> IslandSampler:
    """Factory function to create island samplers.

    Args:
        programs: Program controller
        strategy: Sampling strategy name

    Returns:
        IslandSampler instance

    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == "uniform":
        return UniformIslandSampler(programs)
    elif strategy == "equal":
        return EqualIslandSampler(programs)
    elif strategy == "proportional":
        return ProportionalIslandSampler(programs, temperature=1.0)
    elif strategy == "weighted":
        return WeightedIslandSampler(
            programs,
            fitness_weight=1.0,
            count_weight=1.0,
        )
    else:
        raise ValueError(
            f"Unknown island sampling strategy: {strategy}. "
            f"Valid options: 'uniform', 'equal', 'proportional', 'weighted'"
        )
