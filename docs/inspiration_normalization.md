# Inspiration Normalization Plan

## Goal

Normalize inspirations into a real relationship model while keeping:

- SQLite as the primary store
- `Program` as the main persisted entity
- `Island` as a computed concept
- `RunState` as separate global state
- controllers and search policies as plain Python, not ORM-heavy logic

This is the first major normalization step after stabilizing the repository boundary.

## Why The Current Shape Is Weak

Today inspirations are stored on the child program as JSON arrays:

- `archive_inspiration_ids: list[str]`
- `top_k_inspiration_ids: list[str]`

This is convenient, but it is weak for a few reasons:

1. It encodes a relationship as an attachment field.
2. It duplicates a graph edge in denormalized form.
3. It is difficult to query cleanly.
4. It mixes prompt provenance with program storage.
5. It makes repository design feel awkward because the relation is hidden.

This was acceptable for bootstrapping, but it is the wrong long-term model.

## What An Inspiration Actually Is

An inspiration is not a property of a program.

It is an edge:

- source program `A`
- child program `B`
- role of the edge
- optional order/weight/metadata

So the real model is:

- `Program` nodes
- `InspirationUse` edges

This is a many-to-many relationship:

- one child can reference many source programs
- one source program can influence many children

## Recommended Persisted Model

## Keep

### `programs`

Primary persisted entity.

This remains the main table.

### `metadata_store`

Global run state store.

This remains separate.

## Add

### `program_inspirations`

Normalized join table representing inspiration edges.

Recommended columns:

- `id`
- `child_program_id`
- `source_program_id`
- `role`
- `order_index`
- `weight` nullable
- `edge_metadata` JSON nullable

Recommended constraints/indexes:

- foreign key `child_program_id -> programs.id`
- foreign key `source_program_id -> programs.id`
- index on `child_program_id`
- index on `source_program_id`
- index on `(child_program_id, role, order_index)`
- unique constraint on `(child_program_id, source_program_id, role, order_index)`

## Role Model

Keep roles explicit.

Recommended role enum/string values:

- `archive`
- `top_k`
- `ancestor`

Possible future roles:

- `crossover`
- `manual_seed_reference`
- `meta_reference`

Do not create separate columns for each role. Use one table plus a role field.

## ORM Model Shape

Recommended SQLAlchemy persistence model:

```python
class ProgramInspirationRecord(Base):
    __tablename__ = "program_inspirations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    child_program_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_program_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
```

Important point:

- the ORM model is persistence only
- policies/controllers should still work with plain domain objects

## Domain Model Shape

Add a plain domain object:

```python
@dataclass(frozen=True)
class InspirationUse:
    child_program_id: str
    source_program_id: str
    role: str
    order_index: int = 0
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

This keeps the repository/domain boundary clean.

## Relationship Semantics

### Child-side view

Queries we need:

- what inspired this child?
- in what roles?
- in what order?

### Source-side view

Queries we want:

- which children did this source influence?
- how often is this source reused?
- which source programs produce strong descendants?

This is where normalization pays off immediately.

## What Happens To `Program`

There is no migration burden in this rewrite.

So the clean rule is:

- `Program` may still expose `archive_inspiration_ids` and `top_k_inspiration_ids`
- but those are convenience projections only
- they are reconstructed from `program_inspirations`
- they are not stored on `programs`

That means:

- remove the JSON inspiration-id columns from `programs`
- write only to `program_inspirations`
- read inspiration lists by querying `program_inspirations`

The source of truth becomes:

- `InspirationRepository.list_for_child(program_id)`

## Recommended Repository Surface

Add a dedicated `InspirationRepository`.

### Write methods

- `add_many(child_program_id, inspirations: list[InspirationUse])`
- `replace_for_child(child_program_id, inspirations: list[InspirationUse])`

### Read methods

- `list_for_child(child_program_id) -> list[InspirationUse]`
- `list_sources_for_child(child_program_id, role: str | None = None) -> list[str]`
- `list_children_for_source(source_program_id, role: str | None = None) -> list[str]`
- `count_usage_by_source(source_program_id) -> int`
- `count_usage_by_role(role: str) -> int`

### Optional convenience methods

- `get_archive_source_ids(child_program_id) -> list[str]`
- `get_top_k_source_ids(child_program_id) -> list[str]`

These should be convenience projections only.

## Impact On `ProgramRepository`

`ProgramRepository` should not absorb all inspiration behavior forever.

The right split is:

- `ProgramRepository` owns `Program` CRUD
- `InspirationRepository` owns program-to-program inspiration edges

However, `ProgramRepository.add(program)` may still coordinate with `InspirationRepository` during transition.

Recommended write path:

### `ProgramWriteService` coordinates both

- persist program row
- persist inspiration edges

If that feels too large for the first cut, `ProgramRepository.add(...)` can
temporarily coordinate with `InspirationRepository`, but there should be no
dual-write to legacy JSON fields.

## Query Improvements We Get

With a normalized relation, these become easy:

- find most reused inspiration sources
- find which archive inspirations actually lead to winning children
- compare `archive` vs `top_k` effectiveness
- inspect all descendants influenced by a given source
- compute source-program reuse statistics
- build better lineage + influence visualizations

These queries are awkward or inefficient with JSON arrays.

## SQLite Suitability

SQLite is still the right default.

Why:

- highly portable
- simple operationally
- excellent for local runs and artifact directories
- perfectly capable of supporting normalized join tables at this scale
- integrates cleanly with SQLAlchemy

This normalization does not require a different database.

## Alternative Storage Options

These are worth considering only if requirements change.

### PostgreSQL

Use if:

- you need heavier concurrent write throughput
- you want remote multi-user access
- you want server-managed persistence

Pros:

- better concurrency
- stronger operational scalability

Cons:

- more operational overhead
- worse portability for per-run artifact bundles

### DuckDB

Use if:

- analytics becomes more important than transactional runtime operations

Pros:

- excellent analytical querying

Cons:

- less natural as the primary runtime mutation store here

### Graph database

Do not use now.

Yes, the lineage/inspiration graph is real, but:

- it adds too much operational complexity
- the current scale does not justify it
- SQLite with normalized edge tables is enough

## Recommended Storage Decision

Stay on:

- SQLite
- SQLAlchemy ORM for persisted tables

Normalize:

- `program_inspirations`

Keep computed:

- `Island`

Keep separate:

- `RunState`

Keep plain Python:

- search policies
- controllers
- novelty logic

## Recommended Immediate Implementation Order

1. Add `ProgramInspirationRecord` to `models.py`
2. Add `InspirationUse` domain object
3. Add `InspirationRepository`
4. Remove inspiration-id JSON columns from `ProgramRecord`
5. Persist inspirations only via `program_inspirations`
6. Reconstruct `archive_inspiration_ids` / `top_k_inspiration_ids` on reads if the domain object still needs them
7. Update UI/detail/query helpers to read from `InspirationRepository`

## Big Architectural Point

This normalization is not just “better schema hygiene.”

It improves the actual system model:

- `Program` remains the node
- inspirations become edges
- repositories become more honest
- policies operate over real relationships instead of list blobs

That is exactly the direction the rest of the refactor needs.
