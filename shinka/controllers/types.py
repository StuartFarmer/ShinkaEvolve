from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class InspirationUse:
    child_program_id: str
    source_program_id: str
    role: str
    order_index: int = 0
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Island:
    island_idx: int
    total_programs: int = 0
    correct_programs: int = 0
    best_program_id: Optional[str] = None
    best_score: float = 0.0

    @property
    def initialized(self) -> bool:
        return self.correct_programs > 0


@dataclass(frozen=True)
class RunMetadataSnapshot:
    last_iteration: int = 0
    best_program_id: Optional[str] = None
    beam_search_parent_id: Optional[str] = None
    best_score_generation: int = 0
    best_score_ever: Optional[float] = None
    initial_program_count_adjustment: int = 0


@dataclass(frozen=True)
class ProgramCountSnapshot:
    count: int
    max_timestamp: float | None
