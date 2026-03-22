# Shinka Data Model

## Goal

Define the target data model for the cleaned-up Shinka architecture.

The key design rule is:

- `Program` is the code artifact
- other concerns point to `Program`
- runtime state is separate
- islands remain computed
- prompts and model usage are typed structures
- metadata is last-resort attachment space, not a dump for core fields

This is the model the storage layer and repositories should converge toward.

## Core Principles

### 1. Keep `Program` small

A `Program` should represent the actual code artifact, not every event and
attachment around it.

### 2. Normalize relationships

If something is a relationship, store it as a relationship:

- parent/child
- inspirations
- prompt usage
- model usage

Do not hide these in JSON lists on a flat row.

### 3. Use typed attachments for repeated structures

Prompts, evaluation results, model usage, and proposal details should be typed
objects/models, not arbitrary metadata blobs.

### 4. Keep islands computed

An island is still a derived subpopulation built from `Program.island_idx`.
No separate `islands` table is needed yet.

### 5. Keep run/project state separate

Global search state is not program data.

Examples:

- last iteration
- best program
- beam-search sticky parent
- stagnation counters / best score generation

That belongs in `RunState`, not on a program row and not in program metadata.

## Main Entities

## 1. Project

Represents the overall run/problem space stored in one SQLite database.

This can stay implicit if one DB equals one project, but conceptually it exists.

Suggested fields:

- `id`
- `name`
- `task_name`
- `created_at`
- `config_snapshot_json`

If kept implicit for now, `RunState` can still be modeled as "the state for the
current project".

## 2. Program

The primary evolved artifact.

Suggested fields:

- `id`
- `name`
- `code`
- `language`
- `parent_program_id`
- `island_idx`
- `generation`
- `created_at`
- `system_prompt_id`

Optional but still reasonable as first-class fields:

- `code_diff`
- `children_count`

That is the artifact and lineage shell. It should not directly own all
evaluation, cost, prompt, and run-state concerns.

### Program relationships

- parent: many-to-one to `Program`
- children: one-to-many from `Program`
- inspirations: many-to-many to `Program`
- evaluation: one-to-one to `ProgramEvaluation`
- proposal/origin: one-to-one to `ProgramProposal`
- model usages: one-to-many to `ModelUsage`
- prompt snapshot: one-to-one to `PromptSnapshot`

## 3. ProgramEvaluation

Represents the evaluation/execution result for a program.

Current Shinka behavior is effectively one evaluation per program, so this can
be one-to-one with `Program`.

Suggested fields:

- `program_id`
- `correct`
- `combined_score`
- `text_feedback`
- `compute_time_seconds`
- `public_metrics_json`
- `private_metrics_json`
- `diagnostics_json`

### About public/private metrics

These are evaluator-defined outputs.

They should be treated as typed artifacts stored in JSON:

- public metrics: things the system may use for ranking/display
- private metrics: evaluator-specific hidden details

Because they are task-defined, they should not explode into many hardcoded
columns unless a particular metric becomes structurally important.

## 4. ProgramProposal

Represents how a program was generated.

This is distinct from the program artifact and from the evaluation result.

Suggested fields:

- `program_id`
- `patch_type`
- `patch_name`
- `code_diff`
- `system_prompt_id`
- `proposal_metadata_json`

This is where proposal-side provenance belongs.

## 5. ProgramInspiration

Normalized many-to-many relationship between programs.

This replaces the old "IDs stuffed into JSON arrays" approach.

Suggested fields:

- `child_program_id`
- `source_program_id`
- `role`
- `order_index`
- `weight` optional
- `edge_metadata_json` optional

Suggested roles:

- `archive`
- `top_k`
- `ancestor`
- `fix_source`

This is the correct place for inspirations.

## 6. Prompt

A reusable prompt artifact.

Suggested fields:

- `id`
- `text`
- `role`
- `tag`
- `created_at`
- `metadata_json`

Examples of `role`:

- `system`
- `user`
- `meta`
- `novelty`
- `evaluation`

Examples of `tag`:

- `default`
- `island_family`
- `evolved`
- `diagnostics`

## 7. PromptSnapshot

The exact prompt bundle used for a specific program event.

This is different from reusable prompt definitions.

Suggested fields:

- `id`
- `program_id`
- `system_prompt_id` nullable
- `user_prompt_id` nullable
- `meta_prompt_id` nullable
- `created_at`
- `metadata_json`

This allows:

- reuse of prompt definitions
- exact replayability of what was used for one program

If needed later, this can be generalized into:

- `PromptSnapshot`
- `PromptSnapshotPart`

But the simpler one-row snapshot is enough for now.

## 8. ModelUsage

Uniform record of model usage and costs.

This is better than scattering many ad hoc cost fields across metadata.

Suggested fields:

- `id`
- `program_id`
- `role`
- `model_name`
- `provider`
- `api_cost`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `metadata_json`

Suggested roles:

- `generation`
- `embedding`
- `novelty`
- `meta`

This is the reusable OOP-style structure for all model-related accounting.

## 9. RunState

Typed global state for the project/run.

This replaces "magic metadata keys in a string store" as the conceptual model.

Suggested fields:

- `project_id` or implicit singleton key
- `last_iteration`
- `best_program_id`
- `beam_search_parent_id`
- `best_score_generation`
- `best_score_ever`
- `initial_program_count_adjustment`

This is not program data.

## Computed Concepts

## Island

`Island` remains computed.

It is derived from programs grouped by `island_idx`.

Suggested computed fields:

- `island_idx`
- `total_programs`
- `correct_programs`
- `best_program_id`
- `best_score`
- `initialized`

This should be built by `IslandRepository` or `IslandService` from programs.

No separate `islands` table is necessary yet.

## What Should Stay In Metadata

Metadata should hold only sparse attachments that are not worth promoting to
first-class structured fields.

Good metadata candidates:

- temporary debug notes
- novelty explanation text
- ad hoc evaluator extras
- one-off experiment annotations

## What Should Not Stay In Metadata

These should not be buried in metadata:

- correctness
- combined score
- prompt usage
- inspiration edges
- model costs/usages
- parent relationships
- run state
- patch type
- any value that is queried often or affects behavior

## Recommended Minimal Persistence Layout

Keep the first real normalized schema small:

1. `programs`
2. `program_evaluations`
3. `program_inspirations`
4. `prompts`
5. `prompt_snapshots`
6. `model_usages`
7. `run_state`

`Island` remains computed, not stored.

## Practical Repository Boundaries

Recommended repositories/services:

- `ProgramRepository`
- `ProgramEvaluationRepository`
- `InspirationRepository`
- `PromptRepository`
- `PromptSnapshotRepository`
- `ModelUsageRepository`
- `RunStateRepository`
- `IslandRepository` or `IslandService` for computed island views

## Why This Model Is Better

This model:

- keeps `Program` focused on the artifact
- separates evaluation from generation provenance
- makes inspirations first-class
- makes prompts reusable and replayable
- makes model usage uniform
- keeps run state out of program data
- reduces the need for giant flat tables and metadata dumping

That is the target direction for the remaining cleanup.
