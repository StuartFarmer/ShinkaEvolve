"""Island sampling strategies for parent selection."""

import logging
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from . import island_ops
from .connection import Database
from .types import Island

logger = logging.getLogger(__name__)


class IslandSampler(ABC):
    """Abstract base class for island sampling strategies."""

    def __init__(self, db: Database):
        self.db = db

    @abstractmethod
    def sample_island(self, initialized_islands: List[Island]) -> int:
        pass

    def _normalize_islands(self, initialized_islands: List[Island | int]) -> List[Island]:
        with self.db.session() as session:
            island_map = {
                island.island_idx: island
                for island in island_ops.list_islands(
                    session,
                    num_islands=self.db.num_islands,
                )
            }
        normalized: List[Island] = []
        for island in initialized_islands:
            if isinstance(island, Island):
                normalized.append(island)
            else:
                normalized.append(island_map[int(island)])
        return normalized


class UniformIslandSampler(IslandSampler):
    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        initialized_islands = self._normalize_islands(initialized_islands)
        return random.choice(initialized_islands).island_idx


class EqualIslandSampler(IslandSampler):
    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        initialized_islands = self._normalize_islands(initialized_islands)
        min_count = min(island.correct_programs for island in initialized_islands)
        islands_with_min = [
            island.island_idx
            for island in initialized_islands
            if island.correct_programs == min_count
        ]
        sampled = random.choice(islands_with_min)
        logger.debug(
            "EqualIslandSampler: Island counts = %s, min_count = %s, sampled = %s",
            [island.correct_programs for island in initialized_islands],
            min_count,
            sampled,
        )
        return sampled


class ProportionalIslandSampler(IslandSampler):
    def __init__(self, db: Database, temperature: float = 1.0):
        super().__init__(db)
        self.temperature = temperature

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        initialized_islands = self._normalize_islands(initialized_islands)
        fitness_values = np.array([island.best_score for island in initialized_islands])
        exp_values = np.exp(fitness_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx].island_idx
        logger.debug(
            "ProportionalIslandSampler: fitness = %s, probabilities = %s, sampled = %s",
            [island.best_score for island in initialized_islands],
            probabilities,
            sampled_island,
        )
        return sampled_island


class WeightedIslandSampler(IslandSampler):
    def __init__(self, db: Database, fitness_weight: float = 1.0, count_weight: float = 1.0):
        super().__init__(db)
        self.fitness_weight = fitness_weight
        self.count_weight = count_weight

    def sample_island(self, initialized_islands: List[Island | int]) -> int:
        initialized_islands = self._normalize_islands(initialized_islands)
        weights = []
        for island in initialized_islands:
            count = island.correct_programs or 1
            fitness = island.best_score
            weight = (fitness**self.fitness_weight) / (count**self.count_weight)
            weights.append(weight)
        weights = np.array(weights)
        if np.sum(weights) == 0:
            probabilities = np.ones(len(weights)) / len(weights)
        else:
            probabilities = weights / np.sum(weights)
        sampled_idx = np.random.choice(len(initialized_islands), p=probabilities)
        sampled_island = initialized_islands[sampled_idx].island_idx
        logger.debug(
            "WeightedIslandSampler: counts = %s, fitness = %s, weights = %s, probabilities = %s, sampled = %s",
            [island.correct_programs for island in initialized_islands],
            [island.best_score for island in initialized_islands],
            weights,
            probabilities,
            sampled_island,
        )
        return sampled_island


def create_island_sampler(db: Database, strategy: str = "uniform") -> IslandSampler:
    if strategy == "uniform":
        return UniformIslandSampler(db)
    if strategy == "equal":
        return EqualIslandSampler(db)
    if strategy == "proportional":
        return ProportionalIslandSampler(db, temperature=1.0)
    if strategy == "weighted":
        return WeightedIslandSampler(db, fitness_weight=1.0, count_weight=1.0)
    raise ValueError(
        f"Unknown island sampling strategy: {strategy}. "
        "Valid options: 'uniform', 'equal', 'proportional', 'weighted'"
    )
