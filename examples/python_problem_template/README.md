# Python Problem Template

This is a generic Shinka task scaffold for a Python optimization problem.

Files:

- `initial.py`: the candidate program that Shinka evolves
- `evaluate.py`: the evaluator that scores candidate programs
- `run_evo.py`: optional Python launcher
- `shinka.yaml`: optional config for `run_evo.py`

## What To Fill In

1. Edit `initial.py`
   - Keep the function signatures stable.
   - Put the mutable algorithm inside the `EVOLVE-BLOCK` region.
   - Return `(score, text_feedback)` from `run_experiment(...)`.

2. Edit `evaluate.py`
   - Update `get_experiment_kwargs(...)` to generate your inputs/seeds.
   - Update `validate_result(...)` to enforce correctness.
   - Update `aggregate_metrics(...)` so `combined_score` matches your objective.

3. Edit `run_evo.py`
   - Replace `TASK_SYS_MSG` with a task-specific prompt.
   - Adjust models, budgets, and concurrency if needed.

## Manual Smoke Test

From repo root:

```bash
python examples/python_problem_template/evaluate.py \
  --program_path examples/python_problem_template/initial.py \
  --results_dir /tmp/shinka_python_problem_template_smoke
```

## Run With `shinka_run`

From repo root:

```bash
shinka_run \
  --task-dir examples/python_problem_template \
  --results_dir results/python_problem_template \
  --num_generations 20 \
  --set job.activate_script=.venv/bin/activate
```

## Notes

- Higher `combined_score` is better.
- Only code inside `EVOLVE-BLOCK-START/END` should be mutable.
- If your task needs files or datasets, keep the evaluator contract stable.
