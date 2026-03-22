# Shinka Domain Model

## Goal

Define the conceptual model that should drive the persistence and IO layer.

The main rule is:

- the data model should reflect the real search concepts
- repository methods should be designed from the queries we actually need
- services/controllers should orchestrate behavior over those concepts
- opaque metadata should be minimized for anything that affects search behavior

This document is the target model for continuing the `ProgramDatabase` breakup.

## The Problem With The Current Shape

Today the system works, but the model is still too implementation-shaped:

- `Program` carries too many concerns
- `island_idx` is only an integer tag on a program
- inspirations are stored as JSON lists on the child program
- `metadata` is a mixed bag of:
  - real runtime facts
  - debug/logging details
  - search control hints
  - evaluator artifacts
- some relationships are first-class, others are hidden in JSON

That causes:

- awkward repositories
- too many small adapter classes
- too much logic to reconstruct meaning from raw fields
- difficulty deciding what belongs in storage vs policy

## Core Domain Concepts

These are the actual concepts in the system.

### 1. Program

A `Program` is the primary evolved artifact.

It is the source of truth for:

- code
- correctness
- score
- generation
- parent relationship
- island membership
- evaluator outputs

This is the one truly primary entity.

### 2. Lineage

Lineage is the parent/child graph between programs.

This is not a separate entity row right now; it is represented by:

- `program.parent_id`
- `children_count`

Conceptually:

- every non-root program has zero or one parent
- every program can have many children

This relationship is central and should remain first-class.

### 3. Island

An island is a computed subpopulation.

It is not a first-class persisted row today, and it does not need to be yet.

Conceptually an island is:

- a group of programs with the same `island_idx`
- plus derived aggregate facts such as:
  - population
  - number of correct programs
  - best score
  - best program
  - initialized/not initialized

So:

- `island_idx` on `Program` is the membership fact
- `Island` is the computed domain object over that membership

### 4. Inspiration Use

An inspiration is not a standalone entity.

It is a relationship:

- a child program was proposed while referencing another program
- in a specific role

The roles currently are:

- archive inspiration
- top-k inspiration
- possibly ancestor inspiration in fix mode

This should be modeled as a relationship, not as a free-floating object.

### 5. Run State

Run state is global state about the evolution process, not about any one program.

Examples:

- `last_iteration`
- `best_program_id`
- `beam_search_parent_id`
- `best_score_generation`
- `best_score_ever`
- `initial_program_count_adjustment`

This is not `Program` data and should not be stored on programs.

### 6. Program Analysis

This is structured analysis or evaluator output attached to a program.

Examples:

- `public_metrics`
- `private_metrics`
- `text_feedback`
- complexity/code analysis
- embedding features

Some of this belongs on the program directly because it is queried often.
Some of it can remain attached/secondary.

## Proposed Entity Model

This is the recommended conceptual model.

### Program

Primary persisted entity.

Recommended first-class fields:

- `id`
- `code`
- `language`
- `parent_id`
- `generation`
- `timestamp`
- `correct`
- `combined_score`
- `children_count`
- `island_idx`
- `code_diff`
- `complexity`
- `system_prompt_id`

Recommended first-class analysis fields:

- `public_metrics`
- `private_metrics`
- `text_feedback`
- `embedding`
- `embedding_pca_2d`
- `embedding_pca_3d`
- `embedding_cluster_id`

These are already query-relevant enough to remain first-class.

### Island

Computed domain object, not necessarily a table.

Fields:

- `island_idx`
- `total_programs`
- `correct_programs`
- `best_program_id`
- `best_score`
- derived `initialized`

This should be built by `IslandRepository` from programs.

### InspirationUse

Relationship object, ideally normalized later.

Fields:

- `child_program_id`
- `source_program_id`
- `role`
- `order_index`

Recommended roles:

- `archive`
- `top_k`
- `ancestor`

Today this is encoded as:

- `archive_inspiration_ids: list[str]`
- `top_k_inspiration_ids: list[str]`

That is serviceable, but not the clean target.

### RunState

Global persisted key/value or structured object.

Fields:

- `last_iteration`
- `best_program_id`
- `beam_search_parent_id`
- `best_score_generation`
- `best_score_ever`
- `initial_program_count_adjustment`

This is already conceptually correct as a dedicated store.

## What Should Stay In Metadata

`metadata` should be for attachments, logs, and auxiliary details that are not primary query keys.

Good candidates for metadata:

- raw stdout/stderr logs
- LLM request/response details
- novelty explanation text
- per-run debug info
- patch name labels
- ad hoc evaluator artifacts that are not used for selection/querying

## What Should Not Stay In Metadata

If a value affects search logic, display logic, or common filtering, it should not be buried in metadata.

These should stay first-class:

- `parent_id`
- `generation`
- `correct`
- `combined_score`
- `children_count`
- `island_idx`
- `system_prompt_id`

Likely future candidates to promote:

- `patch_type`
- `model_name`
- `api_cost`
- maybe a normalized `family_id`

Those are currently metadata-heavy and may deserve promotion if they become major query dimensions.

## Relationships

The model should acknowledge these relationships explicitly.

### Program -> Parent Program

- one-to-zero-or-one
- stored by `parent_id`

Used for:

- lineage
- ancestry lookup
- beam/winner refinement context
- tree visualizations

### Program -> Child Programs

- one-to-many
- derived from parent links

Used for:

- subtree copying
- mutation count / branching
- lineage analysis

### Program -> Island

- many-to-one membership
- stored by `island_idx`

Used for:

- local selection
- migration
- island summaries

### Program -> Inspirations

- many-to-many with role

Used for:

- prompt provenance
- later auditability
- inspiration-aware analysis

Current encoding is lossy and should eventually become a normalized link.

### Program -> Prompt

- many-to-zero-or-one
- via `system_prompt_id`

Used for:

- prompt evolution attribution
- prompt fitness analysis

## Required Query Surface

These queries should drive repository design.

### Program queries

- get program by id
- list all programs
- list correct programs
- list programs by generation
- list programs by island
- list top programs by score
- list top programs by a public metric
- get children of a program
- get ancestry of a program
- get embeddings for programs in an island

### Island queries

- list islands
- get one island
- list initialized islands
- get island populations
- get next island index
- get best program in island
- get initial seed/root program

### Run state queries

- get/set last iteration
- get/set best program
- get/set beam-search parent
- get/set best-score generation

### Inspiration queries

Current minimum:

- get archive inspirations for a sampled context
- get top-k inspirations for a sampled context

Future normalized version:

- list inspirations for a child program by role
- list children influenced by a source program

## Repository Implications

The CRUD/IO layer should map to these concepts directly.

### ProgramRepository should own

- `add(program)`
- `get(program_id)`
- `get_many(program_ids)`
- `list_all()`
- `list_correct()`
- `list_by_generation(generation)`
- `list_by_island(island_idx, correct_only=False)`
- `list_top(...)`
- `get_children(program_id)`
- `get_ancestry(program_id)`
- `update_program_metadata(program_id, metadata)`
- `migrate_program(program_id, target_island_idx, migration_record)`
- `list_embeddings_by_island(island_idx)`

It should not own:

- parent selection
- archive policy
- island selection
- novelty judgment

### IslandRepository should own

- computed island queries from program membership
- island aggregate facts

It should not own:

- migration policy
- island selection policy

### MetadataRepository should own

- global run state only

### Future InspirationRepository should own

If inspiration links are normalized:

- create/list `InspirationUse`
- query inspirations by child
- query influenced children by source

## Recommended Near-Term Changes

These are the highest-value model-driven refactors.

### 1. Keep `Program` as the primary domain object

This remains correct.

### 2. Keep `Island` computed, not persisted

No island table is needed yet.

The current computed-island direction is good.

### 3. Introduce a normalized inspiration relationship later

This is the biggest current modeling gap.

Today:

- archive/top-k inspiration ids live as JSON arrays on `Program`

Better:

- `ProgramInspirationLink`

This will simplify prompt provenance, querying, and later analytics.

### 4. Reduce `metadata` usage for operational facts

If a value is used in:

- filtering
- ranking
- grouping
- analysis UI

it should be a real field.

### 5. Design runner/runtime from repositories over this model

The target runtime should be:

- controller
- repositories
- policy/services

not:

- controller -> god database object

## Recommended Long-Term Table Layout

This is the likely clean persistence shape.

### Required now

- `programs`
- `metadata_store`

### Likely next

- `program_inspirations`

### Optional later

- `program_events`
- `program_artifacts`
- `program_logs`

No `islands` table is required unless islands later need persistent identity beyond `island_idx`.

## Practical Decision Rules

Use these rules during refactor.

### Add a first-class field when:

- it is queried often
- it affects search behavior
- it affects ranking/filtering
- it is shown in main UI tables

### Leave it in metadata when:

- it is debug-only
- it is sparse/ad hoc
- it is only used for drill-down inspection

### Add a new repository when:

- there is a distinct persisted concept with its own query surface

### Add a service instead of a repository when:

- the behavior is computed from existing persisted facts

Examples:

- archive is a policy/service
- novelty is a service
- island selection is a service

## The Big Architectural Point

The model should be:

- `Program` as the primary persisted artifact
- `Island` as computed population structure
- `RunState` as global persisted control state
- `InspirationUse` as a relationship

Once the data model is explicit, the rest gets simpler:

- repositories become smaller and clearer
- fewer shims are needed
- selection logic has cleaner inputs
- `ProgramDatabase` becomes unnecessary rather than merely inconvenient
