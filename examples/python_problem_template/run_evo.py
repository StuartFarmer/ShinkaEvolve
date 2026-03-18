#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running directly from a source checkout without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """You are optimizing a Python program for a user-defined problem.

Goal:
- Maximize combined_score.

Rules:
- Only modify code inside EVOLVE-BLOCK markers.
- Preserve function names and external I/O contracts.
- Keep the program correct.

Replace this prompt with your exact task, constraints, and algorithm hints.
"""


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG

    evo_config = EvolutionConfig(**config["evo_config"])
    db_config = DatabaseConfig(**config["db_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        activate_script=config.get("job_config", {}).get("activate_script"),
        conda_env=config.get("job_config", {}).get("conda_env"),
        time=config.get("job_config", {}).get("time"),
    )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        verbose=bool(config.get("verbose", True)),
        debug=bool(config.get("debug", False)),
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="shinka.yaml")
    args = parser.parse_args()
    main(args.config_path)
