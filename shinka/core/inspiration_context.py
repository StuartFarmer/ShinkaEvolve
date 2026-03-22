from __future__ import annotations

from typing import Any, Literal


class InspirationContextBuilder:
    """Helper for ordering inspiration programs before prompt rendering."""

    def __init__(
        self,
        sort_order: Literal["ascending", "chronological", "none"] = "ascending",
    ):
        self.sort_order = sort_order

    def build(self, archive_inspirations: list[Any], top_k_inspirations: list[Any]) -> list[Any]:
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
