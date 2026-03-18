# Two-Phase Hierarchical Exploration Plan

## Summary

Add a new hierarchy-aware orchestration layer in front of Shinka instead of forcing pre-code ideas into the existing `Program` lifecycle.

The system will run in two explicit phases:

1. **Phase 1: Conceptual creation**
   - Generate high-level concepts from the task prompt.
   - Merge similar concepts into families.
   - Materialize a small number of runnable seed programs per family.

2. **Phase 2: Family testing and refinement**
   - Run short Shinka refinement batches per family to test whether the idea family has real signal.
   - Rank families by measured performance.
   - Allocate the remaining budget to the best families for deeper refinement.

This keeps Shinka's strongest behavior, executable testing and iterative improvement, while moving pre-code exploration into a separate first-class layer.

## Implementation Changes

### New orchestration layer

Add a new runner, `HierarchicalExploreRunner`, as a separate subsystem rather than modifying `ProgramDatabase` to hold concepts.

Responsibilities:
- own the two-phase workflow
- manage concept/family state
- create runnable seeds
- launch and monitor family-local Shinka runs
- compare families and promote or terminate them

Keep the existing Shinka async runner unchanged as the refinement engine for family-local runs.

### New domain model

Introduce separate objects for pre-code search:
- `Concept`: raw conceptual candidate produced by the planner LLM
- `Family`: canonicalized group of related concepts with a shared hypothesis
- `FamilySeed`: runnable initial program generated from a family
- `FamilyRun`: one Shinka refinement batch attached to a family
- `FamilyScore`: aggregated evaluation summary used for promotion

Do **not** store these in the existing `Program` table. Persist them in a separate hierarchical-state store under the top-level results directory. Use a separate SQLite database, `hierarchical.sqlite`, for resumability and clean separation from `programs.sqlite`.

### Phase 1 behavior

Phase 1 is text-first, not code-first.

Default flow:
- take the task description and current evaluator contract as input
- generate `num_concepts=12` raw concepts
- deduplicate and merge them into at most `max_families=6` families
- generate `seeds_per_family=2` runnable seeds per family
- validate each seed by running the normal evaluator once before promotion to Phase 2
- discard seeds that fail to execute or produce unusable outputs
- if a family has zero valid seeds, mark the family inactive before Phase 2

Concept generation prompts must explicitly request:
- mechanism
- expected advantage
- failure mode
- what would make the idea meaningfully different from existing families

Family consolidation must explicitly produce:
- family name
- family hypothesis
- family constraints
- distinctiveness criteria used to avoid duplicates

Seed materialization must output executable initial programs, not patches.

### Phase 2 behavior

Run one independent Shinka refinement track per family.

Execution model:
- create per-family results roots under `results/<root>/families/<family_id>/`
- run Shinka with each family seed as the `initial` program
- keep each family's refinement state isolated from other families during warmup

Budget policy:
- `family_warmup_generations=10` per seed
- aggregate warmup results at the family level using:
  - best correct `combined_score`
  - score improvement from seed baseline
  - correctness rate across warmup runs
- promote `surviving_families=2` families after warmup
- allocate the remaining refinement budget only to promoted families
- continue refinement using additional Shinka generations inside each promoted family track

Family ranking rule:
- primary: highest best correct `combined_score`
- secondary: highest median improvement over family seeds
- tertiary: highest correctness rate
- tie-breaker: lower total API cost

### Interface and API additions

Add a new Python entrypoint:
- `HierarchicalExploreRunner(...)`

Add a new CLI entrypoint:
- `shinka_hierarchical_run`

Add a dedicated config block, `hierarchical`, with these defaults:
- `enabled: false`
- `num_concepts: 12`
- `max_families: 6`
- `seeds_per_family: 2`
- `family_warmup_generations: 10`
- `surviving_families: 2`
- `family_isolation: true`
- `concept_model: <defaults to llm_models[0]>`
- `seed_model: <defaults to llm_models[0]>`

Treat the existing Shinka `evo`, `db`, and `job` configs as the per-family refinement config. The hierarchical layer only partitions and schedules budget; it does not redefine evaluation semantics.

## Test Plan

Implement the following tests:

- concept generation returns the requested number of concepts and persists them
- family consolidation merges near-duplicates into a smaller family set deterministically
- seed materialization produces runnable programs for at least one family on a fixture task
- invalid seeds are rejected without crashing the hierarchical run
- warmup promotion selects the correct top families from synthetic family-score fixtures
- family-local Shinka runs write to isolated results directories and do not share DB state
- resume logic restores concepts, families, seeds, family runs, and promotion decisions from `hierarchical.sqlite`
- final summary reports:
  - all families
  - warmup outcomes
  - promoted families
  - best family
  - best final program path

## Assumptions and Defaults

- Phase 1 is genuinely pre-code, so it must not be forced into the existing `Program` abstraction.
- Families are independent during warmup; there is no cross-family inspiration sharing in Phase 2 v1.
- The existing Shinka runner remains the refinement engine and should not be deeply refactored for v1.
- `combined_score` remains the family promotion metric unless the evaluator defines a stronger primary metric later.
- v1 will optimize for clean separation and resumability over tight integration with the current archive/island system.
