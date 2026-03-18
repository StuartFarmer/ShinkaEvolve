from __future__ import annotations

import random
from typing import Any


# EVOLVE-BLOCK-START
def solve_problem_core(problem_input: Any) -> tuple[float, str]:
    """Replace this with the algorithm you want Shinka to improve.

    Return:
    - score: numeric value where higher is better
    - text_feedback: optional evaluator-visible note
    """
    score = 0.0
    text_feedback = (
        "Template baseline. Replace solve_problem_core(...) with your task logic."
    )
    return score, text_feedback


# EVOLVE-BLOCK-END


def solve_problem(problem_input: Any) -> tuple[float, str]:
    """Stable wrapper kept outside the evolve block."""
    return solve_problem_core(problem_input)


def run_experiment(random_seed: int | None = None, **kwargs: Any) -> tuple[float, str]:
    """Entry point used by the Shinka evaluator.

    You can pass any task-specific kwargs from evaluate.py via get_experiment_kwargs().
    """
    if random_seed is not None:
        random.seed(random_seed)

    problem_input = kwargs.get("problem_input")
    score, text_feedback = solve_problem(problem_input)
    return float(score), str(text_feedback)
