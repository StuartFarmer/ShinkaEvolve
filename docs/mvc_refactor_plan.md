# Shinka MVC Refactor Plan

## Goal

Move Shinka toward a cleaner MVC-style architecture:

- Models: ORM persistence models only
- Controllers: the application/query layer used to interact with models
- Views: CLI tables, web UI responses, exports, plots

This replaces the current mixed "repository plus service plus helper" shape
with a simpler mental model.

## Target Architecture

## Models

These are persistence-only SQLAlchemy models:

- `ProjectRecord`
- `ProgramRecord`
- `ProgramEvaluationRecord`
- `ProgramProposalRecord`
- `ProgramEmbeddingRecord`
- `ProgramEmbeddingProjectionRecord`
- `ProgramInspirationRecord`
- `PromptRecord`
- `PromptSnapshotRecord`
- `ModelUsageRecord`
- `RunStateRecord`

Models should not become fat active-record objects. They are schema, not
business logic.

## Controllers

Controllers are the main interaction layer over the models.

Planned controllers:

- `ProgramController`
- `EvaluationController`
- `ProposalController`
- `EmbeddingController`
- `InspirationController`
- `PromptController`
- `PromptSnapshotController`
- `ModelUsageController`
- `RunStateController`
- `IslandController`

Each controller should own coherent interactions with one concept.

## Views

These are output/representation layers:

- rich CLI summaries
- web UI serializers
- dataframe/export helpers
- plots

Views should consume controllers/domain objects, not direct SQL or raw ORM
records.

## Migration Strategy

Do not rewrite everything in one shot.

Use this order:

1. Add controllers around the current ORM/session layer.
2. Move obviously misplaced queries into the right controller.
3. Keep existing repository surfaces as temporary adapters.
4. Update call sites incrementally to use controllers directly.
5. Delete repository methods once no call sites depend on them.

## First Cut

### Ticket 1: Add `ProgramController`

Own program-centric queries like:

- `get_initial_program_row()`
- `get_best_program_row()`
- later:
  - `get_program()`
  - `list_by_generation()`
  - `list_top()`

### Ticket 2: Add `IslandController`

Own only island-scoped computed queries like:

- `get_program_island()`
- `list_islands()`
- `list_initialized_islands()`
- `list_initialized_island_ids()`
- `are_all_islands_initialized()`
- `get_island_populations()`
- `get_program_count()`
- `get_next_island_index()`

Important:

- `get_initial_program_row()` does not belong here
- `get_best_program_row()` does not belong here

### Ticket 3: Delegate A Few Existing Calls

Use controllers behind the current `ProgramRepository` adapter for:

- `get_initial_program_row()`
- `get_best_program_row()`
- `list_islands()`
- `list_initialized_islands()`
- `list_initialized_island_ids()`
- `get_next_island_index()`

This begins the migration without breaking behavior.

### Ticket 4: Move Real Call Sites To Controllers

After the adapter layer is stable, update code to call controllers directly in:

- island runtime logic
- context sampling
- async runner
- UI helpers

### Ticket 5: Delete Misplaced Repository Methods

Once callers are migrated:

- remove those methods from `ProgramRepository`
- remove `IslandRepository` or reduce it to a compatibility shim

## Rules

- Do not move business/query logic onto ORM models.
- Do not create new god-controllers.
- Keep each controller narrow and conceptually coherent.
- Prefer explicit session-backed controller methods over raw cursor SQL.
