# Novelty Judge I/O and Refactor Target

This note documents the novelty path as it exists today in Shinka:

- what goes into the novelty layer
- what comes out
- what downstream code actually uses
- what a cleaner replacement should roughly look like

The goal is not to defend the current design. The goal is to make the current
contracts explicit so the next refactor has a clear migration target.

## Current Scope

The current novelty logic is split across:

- `shinka/core/novelty_judge.py`
- `shinka/core/async_novelty_judge.py`
- `shinka/database/dbase.py`
- `shinka/core/async_runner.py`

Conceptually, the novelty path currently mixes 3 responsibilities:

1. deciding whether novelty checking should run at all
2. computing similarity context for one proposal
3. making an accept/reject novelty decision

Those should be separate, but today they are partially bundled together.

## Current Data Types

### 1. Candidate Proposal Inputs

The novelty path does not receive a rich proposal object. It receives a loose
set of values:

- `exec_fname: str`
  - filesystem path to the generated candidate code file
- `code_embedding: list[float]`
  - embedding of the candidate code
- `parent_program: Program`
  - the program this proposal descended from
- `database: ProgramDatabase`
  - used for similarity lookup
- `generation: int`
  - only used by `should_check_novelty`

There is no dedicated `ProposalCandidate` structure today.

### 2. Existing Program Type

The novelty layer depends on `Program` from `shinka/database/dbase.py`.

Fields most relevant to novelty:

- `id: str`
- `code: str`
- `embedding: list[float]`
- `island_idx: int | None`
- `generation: int`
- `combined_score: float`
- `correct: bool`
- `metadata: dict[str, Any]`

The novelty LLM currently compares the new candidate only against one existing
`Program`: the single nearest embedded neighbor in the parent's island.

### 3. Similarity Data Returned by DB Today

Current low-level DB methods:

- `compute_similarity(code_embedding, island_idx) -> list[float]`
- `get_most_similar_program(code_embedding, island_idx) -> Program | None`

Important limitations:

- `compute_similarity(...)` returns only raw similarity scores
  - no program IDs
  - no `(program, score)` pairs
  - no top-k structure
- `get_most_similar_program(...)` does a separate pass and returns only the
  single nearest program

So the current novelty layer has:

- one all-neighbors score list for thresholding
- one separate nearest-neighbor object for the LLM comparison

There is no first-class `NoveltyContext` object today.

## Current Class Interfaces

### `NoveltyJudge.__init__(...)`

Current constructor inputs:

- `novelty_llm_client: LLMClient | None`
- `language: str`
- `similarity_threshold: float`
- `max_novelty_attempts: int`

Current state held on the class:

- `self.novelty_llm_client`
- `self.language`
- `self.similarity_threshold`
- `self.max_novelty_attempts`

Observations:

- `similarity_threshold` is a policy/control value
- `max_novelty_attempts` is orchestration state, not judging logic
- `language` is prompt formatting state
- only `novelty_llm_client` is truly judge-specific

### `NoveltyJudge.should_check_novelty(...)`

Inputs:

- `code_embedding: list[float]`
- `generation: int`
- `parent_program: Program | None`
- `database: ProgramDatabase`

Output:

- `bool`

Current logic:

- returns `False` if:
  - no embedding
  - generation is `0`
  - no parent
- returns `True` only if:
  - parent has an `island_idx`
  - island manager exists
  - all islands are initialized

This is not judging. It is a controller-side gate predicate.

### `NoveltyJudge.assess_novelty_with_rejection_sampling(...)`

Inputs:

- `exec_fname: str`
- `code_embedding: list[float]`
- `parent_program: Program`
- `database: ProgramDatabase`

Output:

- `tuple[bool, dict]`
  - `bool` = `should_accept`
  - `dict` = `novelty_metadata`

Current internal steps:

1. compute similarity scores against all programs in the parent's island
2. compute `max_similarity`
3. if `max_similarity <= similarity_threshold`, accept
4. otherwise fetch the single most similar program
5. if a novelty LLM exists, compare:
   - `existing_code = most_similar_program.code`
   - `proposed_code = Path(exec_fname).read_text()`
6. parse the LLM result into `is_novel`
7. accept/reject

Important note:

The method name says "rejection sampling", but it does not itself generate a new
candidate. It loops internally, but unless something external changes, it is
re-checking the same candidate. The real rejection sampling behavior is owned by
the outer runner, which generates a new proposal after a rejection.

### `NoveltyJudge.check_llm_novelty(...)`

Inputs:

- `proposed_code: str`
- `most_similar_program: Program`

Output:

- `tuple[bool, str, float]`
  - `bool` = `is_novel`
  - `str` = explanation
  - `float` = API cost

This is the actual semantic judge function.

It is currently a strict 1-to-1 comparison:

- one proposed candidate
- one existing nearest-neighbor program

It is not comparing against a top-k set.

## Current `novelty_metadata` Structure

Today `assess_novelty_with_rejection_sampling(...)` returns:

```python
{
    "novelty_checks_performed": int,
    "novelty_total_cost": float,
    "novelty_explanation": str,
    "max_similarity": float,
    "similarity_scores": list[float],
}
```

### Which fields are actually used downstream?

Used later in `async_runner.py`:

- `novelty_checks_performed`
- `novelty_total_cost`
- `novelty_explanation`

Not used later for decisions:

- `max_similarity`
- `similarity_scores`

Those are currently dead observational payload once the judge returns.

## Current Downstream Flow

In `async_runner.py`, the returned values are used like this:

1. `should_accept` controls immediate flow:
   - `True` => submit for evaluation
   - `False` => reject and resample

2. `novelty_metadata` is partially copied into patch/program metadata:
   - `novelty_checks_performed`
   - `novelty_cost`
   - `novelty_explanation`

3. those values are then:
   - stored in `Program.metadata`
   - shown in UI / cost plots
   - available for inspection

What does **not** currently happen:

- novelty metadata does not later alter patch probabilities
- novelty metadata does not later alter parent selection
- novelty metadata does not later alter archive selection
- novelty metadata does not later alter temperatures / model choice

So today:

- the accept/reject boolean matters operationally
- the metadata mostly matters observationally

## What the Current Design Is Actually Doing

The current novelty system is best described as:

### Layer 1: Gate Predicate

`should_check_novelty(...)`

Question:

- should novelty filtering even run for this candidate?

### Layer 2: Cheap Duplicate Filter

`compute_similarity(...)`

Question:

- is this candidate embedding too close to the current island's neighborhood?

### Layer 3: Semantic Tie-Breaker

`check_llm_novelty(...)`

Question:

- even if the embedding is very close, is the code still meaningfully different?

### Layer 4: Outer Rejection Sampling

Runner loop in `async_runner.py`

Question:

- if this candidate is rejected, should we generate another one?

This layering is reasonable in spirit, but the current class boundaries are not
clean.

## Rough Refactor Target

The cleaner replacement should separate:

1. novelty gating policy
2. novelty context computation
3. novelty decision logic
4. retry/resample orchestration

### Proposed Input Structure

Replace the loose parameter bundle with something like:

```python
@dataclass
class ProposalCandidate:
    exec_fname: str
    code: str
    code_embedding: list[float]
    parent_program: Program
    generation: int
```

### Proposed Context Structure

Instead of returning a flat list of scores plus a separate nearest-neighbor
lookup, compute a first-class novelty context:

```python
@dataclass
class NoveltyNeighbor:
    program_id: str
    program: Program
    similarity: float


@dataclass
class NoveltyContext:
    island_idx: int | None
    max_similarity: float
    nearest_neighbors: list[NoveltyNeighbor]  # top-k sorted desc
    all_similarity_count: int
```

This is the big missing data structure today.

It would give the system:

- exact nearest-neighbor identity
- top-k neighborhood, not just top-1
- density information
- reusable context for both thresholding and LLM judging

### Proposed Decision Structure

Instead of `(bool, dict)`, return a typed result:

```python
@dataclass
class NoveltyDecision:
    should_accept: bool
    reason: str
    checks_performed: int
    total_cost: float
    explanation: str
    max_similarity: float | None
    neighbor_ids: list[str]
```

This avoids the current anonymous metadata dict.

### Proposed Class Decomposition

#### 1. `NoveltyPolicy`

Controller-owned gate logic:

```python
def should_run(candidate: ProposalCandidate, state: SearchState) -> bool:
    ...
```

This should absorb today's `should_check_novelty(...)`.

#### 2. `NoveltyContextProvider`

DB / retrieval layer:

```python
def compute_context(candidate: ProposalCandidate, top_k: int = 5) -> NoveltyContext:
    ...
```

This should replace:

- `compute_similarity(...)`
- `get_most_similar_program(...)`

with a single reusable call.

#### 3. `NoveltyJudge`

Actual semantic decision layer:

```python
def judge(candidate: ProposalCandidate, context: NoveltyContext) -> NoveltyDecision:
    ...
```

This is where subclassing should happen.

Examples:

- `ThresholdOnlyNoveltyJudge`
- `SingleNeighborLLMNoveltyJudge`
- `TopKLLMNoveltyJudge`
- `ClusterAwareNoveltyJudge`

#### 4. Outer Controller / Runner

Owns the retry loop:

```python
for attempt in range(max_novelty_attempts):
    candidate = propose(...)
    if not novelty_policy.should_run(candidate, state):
        accept
    context = context_provider.compute_context(candidate, top_k=5)
    decision = novelty_judge.judge(candidate, context)
    if decision.should_accept:
        accept
    else:
        resample
```

This is where rejection sampling actually belongs.

## Minimal Refactor Recommendation

If doing this incrementally, the smallest sensible steps are:

1. move `should_check_novelty(...)` out of `NoveltyJudge`
2. add a new DB method:
   - `get_nearest_neighbors(code_embedding, island_idx, top_k)`
3. introduce a typed `NoveltyContext`
4. rename `assess_novelty_with_rejection_sampling(...)`
   to something closer to:
   - `evaluate_candidate_novelty(...)`
5. make the outer controller own retries explicitly

That gets the architecture much closer to composable and subclassable without
rewriting the entire runner.

## Practical Summary

Today:

- the actual semantic judge is `check_llm_novelty(...)`
- the novelty class also owns a gate predicate and some orchestration
- the DB exposes scores and nearest-neighbor lookup separately
- the metadata is mostly observational after the immediate accept/reject decision

What we should roughly make:

- one candidate object
- one novelty context object
- one novelty decision object
- one actual judge interface
- outer controller owns retries and policy

That would make novelty:

- composable
- testable
- extensible
- easier to wire into adaptive search behavior later
