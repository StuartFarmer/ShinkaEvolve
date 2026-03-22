# ProgramDatabase Replacement Path

## Goal

Remove `ProgramDatabase` as the central runtime object and replace it with:

- `ProgramRepository` for persisted `Program` storage
- `MetadataRepository` for run-state metadata
- `IslandRepository` for computed island facts
- policy/services for search behavior
- a controller/runner that orchestrates those pieces directly

The target runtime shape is:

`context sampler -> prompt builder -> proposer -> candidate materializer -> embedder -> novelty gate -> evaluator -> program ingestor -> repositories`

## Current State

`ProgramDatabase` is still a compatibility hub. It has already lost some responsibilities:

- archive persistence is no longer a source of truth
- context sampling now routes through repository-backed services
- a lot of read APIs are deprecated in favor of `ProgramRepository`
- island CRUD has been pushed into `ProgramRepository`

But it still owns too much.

## What ProgramDatabase Still Does

### 1. Connection bootstrap

It still owns:

- SQLite connection creation
- schema creation/migration
- WAL pragmas
- read-only vs read-write connection setup

This should move to a smaller storage/bootstrap layer.

### 2. Program write orchestration

It still owns:

- `add(program, verbose=False)`
- parent `children_count` side effects
- best-program tracking
- generation tracking
- expensive post-write operations like recompute/scheduled checks

This should be split into:

- `ProgramRepository.add(...)`
- `ProgramIngestor`
- optional post-write hooks/services

### 3. Compatibility read surface

It still exposes wrappers such as:

- `get(...)`
- `get_best_program(...)`
- `sample(...)`
- `sample_with_fix_mode(...)`
- various summary/list methods

These should go away once callers move to repository/services directly.

### 4. Island scheduling hooks

It still coordinates:

- `check_and_spawn_island_if_stagnant(...)`
- `check_scheduled_operations()`
- calling the island manager after writes

This should move into a controller-side orchestration layer.

### 5. Embedding / clustering recomputation path

It still owns:

- `_recompute_embeddings_and_clusters()`
- related persistence updates

This should become a separate service, not a DB object concern.

## Remaining Callers

As of now, the main remaining dependency clusters are:

### Runtime

- `shinka/core/async_runner.py`
- `shinka/database/async_dbase.py`
- `shinka/core/async_summarizer.py`

These are the real blockers. Once they move off `ProgramDatabase`, the class can stop being runtime-critical.

### Public exports / compatibility

- `shinka/database/__init__.py`
- tests importing `ProgramDatabase`
- examples importing `ProgramDatabase`

These can be updated later after runtime is migrated.

## Replacement Strategy

Do not try to delete `ProgramDatabase` in one shot. Replace it in layers.

## Phase 1: Stop Using ProgramDatabase For Reads

Status: mostly done.

Objective:

- all read/list/query/sampling behavior should come from:
  - `ProgramRepository`
  - `MetadataRepository`
  - `IslandRepository`
  - `ContextSampler`
  - search policies

Exit criteria:

- no runtime path depends on `ProgramDatabase.get*`, `sample*`, or archive methods

Notes:

- compatibility wrappers can remain temporarily, but they should not be used by core runtime code

## Phase 2: Extract ProgramIngestor

Objective:

Create a write-side orchestrator that owns:

- persist program through `ProgramRepository`
- update generation / best metadata
- trigger island copy/spawn/migration checks
- trigger expensive post-write work if enabled

Suggested interface:

```python
ingestor.ingest(
    program: Program,
    *,
    verbose: bool = False,
    run_post_write_hooks: bool = True,
) -> str
```

This is the write-side replacement for `ProgramDatabase.add(...)`.

Internally it should depend on:

- `ProgramController`
- `RunStateController`
- `IslandController`
- optional embedding/clustering service

Exit criteria:

- `ProgramDatabase.add(...)` becomes a deprecated shim over `ProgramIngestor`

## Phase 3: Extract Storage Bootstrap

Objective:

Move connection/schema/bootstrap concerns out of `ProgramDatabase`.

Suggested replacements:

- `StorageSession`
- or `RepositoryBundle`

This object should own:

- sqlite connection
- cursor if still needed
- schema initialization
- construction of:
  - `ProgramRepository`
  - `MetadataRepository`
  - `IslandRepository`

Potential shape:

```python
bundle = RepositoryBundle.open(config, read_only=False)
bundle.programs
bundle.metadata
bundle.islands
bundle.close()
```

Exit criteria:

- `ProgramDatabase.__init__` no longer contains unique connection/bootstrap logic

## Phase 4: Move AsyncProgramDatabase Off ProgramDatabase

This is a major step.

Right now `AsyncProgramDatabase` is an async wrapper around `ProgramDatabase`.

That should become an async facade over:

- `ProgramRepository`
- `ProgramIngestor`
- repository bundle/session factory for thread-local read connections

Target rename:

- `AsyncProgramStore`
- or `AsyncProgramService`

It should not require a `ProgramDatabase` instance.

Exit criteria:

- `shinka/database/async_dbase.py` no longer imports or depends on `ProgramDatabase`

## Phase 5: Move AsyncRunner Off ProgramDatabase

This is the decisive runtime migration.

`async_runner.py` should stop holding:

- `self.db: ProgramDatabase`

and instead hold something like:

- `self.repositories`
- `self.programs`
- `self.metadata`
- `self.islands`
- `self.context_sampler`
- `self.program_ingestor`
- `self.async_program_store`

What to replace:

- `self.db.sample()` -> `self.context_sampler.sample(...)`
- `self.db.add(...)` -> `self.program_ingestor.ingest(...)`
- `self.db.get_best_program(...)` -> `self.programs.get_best(...)`
- `self.db.last_iteration` / `self.db.best_program_id` -> metadata snapshot / repository
- `self.db.island_manager...` -> injected island controller/service

Exit criteria:

- `async_runner.py` does not import `ProgramDatabase`

## Phase 6: Remove ProgramDatabase From Threaded Utility Paths

Current thread helpers still instantiate `ProgramDatabase(config)` in several places.

Those should be replaced with:

- `ProgramRepository.from_config(...)`
- `RepositoryBundle.open(...)`
- async/service equivalents

Known remaining patterns:

- `thread_db = ProgramDatabase(self.db.config)`
- `ProgramDatabase(self.sync_db.config, read_only=True)`

These should become repository/session factories instead.

## Phase 7: Deprecate Publicly, Then Delete

Once runtime is migrated:

- keep `ProgramDatabase` as a thin compatibility shim for one transition window
- mark all methods deprecated
- update tests/examples
- remove it from preferred public API exports

Final state:

- `ProgramDatabase` deleted or reduced to a tiny adapter with no unique logic

## Recommended New Core Objects

### RepositoryBundle

Owns:

- connection lifecycle
- schema/bootstrap
- repository construction

### ProgramRepository

Owns:

- all persisted program CRUD
- row serialization
- island-scoped and score-scoped queries

### ProgramIngestor

Owns:

- post-evaluation program persistence workflow
- metadata updates
- island side effects
- optional post-write hooks

### IslandService

Owns:

- assign program island
- copy seed programs
- migration
- spawning

This is now the island-focused controller/service layer rather than a standalone manager object.

### AsyncProgramStore

Owns:

- async/thread-safe repository operations
- no dependency on `ProgramDatabase`

## Concrete Immediate Next Steps

1. Introduce `ProgramIngestor` and move `ProgramDatabase.add(...)` logic into it.
2. Introduce `RepositoryBundle` so bootstrap is no longer embedded in `ProgramDatabase`.
3. Refactor `AsyncProgramDatabase` to depend on repositories/ingestor instead of `ProgramDatabase`.
4. Refactor `AsyncRunner` to hold repositories/services instead of `self.db`.

That is the shortest real path to replacing `ProgramDatabase` completely.

## Non-Goals For This Cut

Do not mix these into the same refactor:

- novelty redesign
- prompt evolution redesign
- meta summarizer redesign
- full async architecture redesign
- adding ORM tables for islands/archive

Those can all happen after `ProgramDatabase` is no longer the god object.
