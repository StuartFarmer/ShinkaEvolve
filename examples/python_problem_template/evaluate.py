from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

# Allow running directly from a source checkout without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shinka.core import run_shinka_eval


def get_experiment_kwargs(run_idx: int) -> dict[str, Any]:
    """Provide per-run inputs.

    Replace this with your actual task inputs, seeds, datasets, or parameters.
    """
    return {
        "random_seed": run_idx + 1,
        "problem_input": {
            "example_id": run_idx,
            "payload": "replace me with your task input",
        },
    }


def validate_result(result: Any) -> tuple[bool, str | None]:
    """Enforce correctness constraints for one run result.

    Expected default result shape: (score, text_feedback)
    """
    if not isinstance(result, tuple) or len(result) != 2:
        return False, "run_experiment(...) must return a 2-tuple: (score, text_feedback)"

    score, text_feedback = result
    if not isinstance(score, (int, float)):
        return False, "The first return value must be numeric"
    if not isinstance(text_feedback, str):
        return False, "The second return value must be a string"
    return True, None


def aggregate_metrics(results: list[Any]) -> dict[str, Any]:
    """Aggregate repeated runs into the Shinka metrics schema."""
    scores = [float(score) for score, _ in results]
    feedback = [text for _, text in results if text]

    combined_score = float(mean(scores)) if scores else 0.0
    return {
        "combined_score": combined_score,
        "public": {
            "num_runs": len(results),
            "mean_score": combined_score,
        },
        "private": {
            "scores": scores,
        },
        "extra_data": {
            "all_feedback": feedback,
        },
        "text_feedback": feedback[0] if feedback else "",
    }


def main(program_path: str, results_dir: str) -> None:
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=3,
        get_experiment_kwargs=get_experiment_kwargs,
        aggregate_metrics_fn=aggregate_metrics,
        validate_fn=validate_result,
    )
    if not correct:
        raise RuntimeError(err or "Evaluation failed")

    print(f"combined_score={metrics.get('combined_score', 0.0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(program_path=args.program_path, results_dir=args.results_dir)
