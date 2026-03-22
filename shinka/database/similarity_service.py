from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from . import embedding_ops, program_reads
from .connection import Database

if TYPE_CHECKING:
    from .program import Program

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarProgram:
    program: "Program"
    similarity: float


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(arr1)
    norm_b = np.linalg.norm(arr2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm_a * norm_b))


class SimilarityService:
    """
    Database-backed embedding similarity queries.

    This owns vector math and nearest-neighbor lookup. It does not own novelty
    policy; callers decide what to do with the returned similarities.
    """

    def __init__(self, db: Database):
        self.db = db

    def compute_similarity(
        self,
        code_embedding: List[float],
        island_idx: int,
    ) -> List[float]:
        if not code_embedding:
            logger.warning("Empty code embedding provided to compute_similarity")
            return []

        with self.db.session() as session:
            rows = embedding_ops.list_by_island(session, island_idx)
        return [
            cosine_similarity(code_embedding, embedding)
            for _, embedding in rows
            if embedding
        ]

    def get_most_similar_program(
        self,
        code_embedding: List[float],
        island_idx: int,
    ) -> Optional["Program"]:
        if not code_embedding:
            logger.warning(
                "Empty code embedding provided to get_most_similar_program"
            )
            return None

        with self.db.session() as session:
            rows = embedding_ops.list_by_island(session, island_idx)
            best: Optional[SimilarProgram] = None
            for program_id, embedding in rows:
                if not embedding:
                    continue
                similarity = cosine_similarity(code_embedding, embedding)
                if best is None or similarity > best.similarity:
                    program = program_reads.get(session, program_id)
                    if program is not None:
                        best = SimilarProgram(program=program, similarity=similarity)
        return None if best is None else best.program
