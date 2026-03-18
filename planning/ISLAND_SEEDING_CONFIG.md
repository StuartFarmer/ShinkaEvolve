# Multi-Family Island Seeding

## What This Adds

Shinka can now accept `evo.island_seeds`, where each seed defines:

- a family identity
- an optional family context
- an optional island-specific system prompt
- an initial runnable program
- an explicit island assignment

Each seeded program is inserted as its own generation-0 island root instead of using the default "copy one initial program to all islands" behavior.

## Seed Shape

Each `evo.island_seeds` entry is a dictionary with:

- `init_program_path`: required path to the initial program file
- `family_id`: optional stable identifier for the family
- `family_name`: optional display name
- `context`: optional family-specific context appended to future prompts
- `task_sys_msg`: optional family-specific system prompt override
- `seed_label`: optional label stored in metadata
- `island_idx`: optional explicit island index; defaults to list order

## Example YAML

```yaml
db_config:
  num_islands: 10

evo_config:
  island_seeds:
    - family_id: linear_rls
      family_name: Online Linear RLS
      island_idx: 0
      init_program_path: seeds/linear_rls.py
      context: >
        Predict next-step return with a linear online model and use
        forgetting to handle regime shifts.
      task_sys_msg: >
        Build stable online linear trading algorithms with strong
        regularization and bounded positions.

    - family_id: mean_variance
      family_name: Mean Variance Control
      island_idx: 1
      init_program_path: seeds/mean_variance.py
      context: >
        Estimate expected return and risk online, then size exposure
        using certainty-equivalent control.
```

## Example CLI Override

```bash
shinka_run \
  --task-dir examples/my_task \
  --results_dir results/trading_families \
  --num_generations 100 \
  --set db.num_islands=10 \
  --set 'evo.island_seeds=[
    {"family_id":"linear_rls","island_idx":0,"init_program_path":"seeds/linear_rls.py","context":"Linear online regression with forgetting."},
    {"family_id":"mean_variance","island_idx":1,"init_program_path":"seeds/mean_variance.py","context":"Risk-aware certainty-equivalent sizing."}
  ]'
```

## Current Limitation

The CLI still validates that the task directory contains a fallback `initial.<ext>` file even when `evo.island_seeds` is used. The actual run will use the island seeds, but the placeholder initial file still needs to exist for now.
