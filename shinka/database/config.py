from dataclasses import dataclass, field
from typing import Dict, Optional

from shinka.defaults import default_archive_criteria


@dataclass
class DatabaseConfig:
    """
    Search-memory and lineage-structure defaults used at the config edge.

    This remains a schema/default carrier for YAML/CLI entrypoints. Runtime
    classes should receive explicit arguments instead of this bag object.
    """

    db_path: Optional[str] = None
    num_islands: int = 2
    archive_size: int = 40
    elite_selection_ratio: float = 0.3
    num_archive_inspirations: int = 1
    num_top_k_inspirations: int = 1
    migration_interval: int = 10
    migration_rate: float = 0.0
    island_elitism: bool = True
    enforce_island_separation: bool = True
    island_selection_strategy: str = "uniform"
    enable_dynamic_islands: bool = False
    stagnation_threshold: int = 100
    island_spawn_strategy: str = "initial"
    island_spawn_subtree_size: int = 1
    parent_selection_strategy: str = "weighted"
    exploitation_alpha: float = 1.0
    exploitation_ratio: float = 0.2
    parent_selection_lambda: float = 10.0
    num_beams: int = 5
    archive_selection_strategy: str = "fitness"
    archive_criteria: Dict[str, float] = field(default_factory=default_archive_criteria)
