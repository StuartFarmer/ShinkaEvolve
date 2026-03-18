# Shinka Workflow And Sampler Guide

## The Short Version

Shinka has more than one thing called a "sampler," and they are not the same:

- The **LLM model sampler** is the actual bandit in `shinka/llm/prioritization.py`.
- The **patch sampler** in `shinka/core/sampler.py` chooses patch format and prompt context. It is not a bandit.
- The **system prompt sampler** used by prompt evolution is a separate UCB-style sampler over prompt variants.

If you mean "the sampler" as in "some sort of bandit," you almost certainly mean the LLM model sampler.

## How The LLM Model Sampler Works

### What it is sampling

Before Shinka asks an LLM to generate a patch, it chooses which model to use from `evo.llm_models`.

That choice is controlled by `evo.llm_dynamic_selection`:
- `ucb` or `ucb1`: `AsymmetricUCB`
- `thompson`: `ThompsonSampler`
- `fixed`: `FixedSampler`
- `None`: no dynamic selection, just use the configured model flow

The default config uses `ucb`.

### When the model is chosen

For each proposal generation, the runner samples the model once before patch generation starts. That sampled model is then used throughout the patch attempt loop for that proposal.

Conceptually:
1. compute posterior probabilities across models
2. sample one model according to that posterior
3. send the patch prompt to that model
4. later, update the sampler with the observed result and cost

### What signal it learns from

The reward is not "did the API call succeed." It is based on the resulting program quality:

- if the child program is correct, reward is based on `combined_score`
- if the child program is incorrect, reward is treated as missing / worst-case
- the reward is shifted relative to a baseline, typically the parent score and/or the current global baseline
- with asymmetric scaling enabled, negative improvement is clipped to zero

So the bandit is learning which model produces useful improvements over the current program, not just which model talks the most.

### `AsymmetricUCB`

This is the default and the most important one to understand.

It keeps track of:
- submitted pulls per model
- completed pulls per model
- accumulated reward
- optional cost observations

Its scoring logic is:
1. estimate a normalized exploitation term from observed rewards
2. add a UCB exploration bonus proportional to `sqrt(log(t) / n)`
3. optionally blend in a cost-aware term so cheaper models get some preference
4. apply epsilon-greedy behavior so non-winning arms still get sampled sometimes

Important behavior:
- unseen models get sampled first
- reward is improvement-oriented, not absolute-output-oriented
- cost can matter if `cost_aware_coef > 0`
- old observations decay over time if auto-decay is enabled

### `ThompsonSampler`

This version maps reward into a 0-1 style success signal and maintains Beta posteriors per model.

Conceptually:
1. convert improvement into a scaled utility
2. update Beta parameters
3. draw a sample per model from the posterior
4. use epsilon-greedy on the sampled winners

This is more stochastic than UCB and can be useful if you want stronger posterior-style exploration.

### `FixedSampler`

This is not really learning. It just samples according to fixed prior probabilities and logs pulls/costs for reporting.

Use it when you want controlled mixing across models without adaptation.

## How The Patch Sampler Works

The patch sampler is separate from the model bandit.

Its job is to construct the actual patch-generation prompt:
- pick patch type: `diff`, `full`, or `cross`
- include the parent program
- include archive inspirations and top-k inspirations
- optionally include one sampled meta recommendation
- order inspirations according to the configured sort order

The patch sampler is doing prompt assembly and mutation-format selection, not online learning.

## How The Prompt Sampler Works

If prompt evolution is enabled, Shinka also maintains a prompt archive and samples system prompts from it using a UCB-style policy.

That sampler is separate from the LLM model bandit:
- model bandit chooses **which model**
- prompt sampler chooses **which system prompt variant**
- patch sampler chooses **which edit style and context**

## End-To-End Shinka Workflow

### 1. Startup and configuration

Shinka initializes:
- results directory
- SQLite program database
- async database wrapper
- job scheduler
- LLM client
- embedding client
- patch sampler
- optional meta summarizer
- optional novelty judge
- optional LLM bandit
- optional prompt evolution database and sampler

If a previous results directory exists, it resumes from the existing database and reloads bandit state.

### 2. Initial program setup

Shinka starts from `initial.py` or generates an initial program with an LLM if no initial file is provided.

Then it:
1. writes the generation-0 program to disk
2. evaluates it
3. embeds it
4. inserts it into the program database

That gives the system a real seed program with actual metrics before evolution begins.

### 3. Island initialization

Programs live on islands.

By default, the initial correct program is copied across islands so each island can start from the same executable root. Until all islands have at least one correct program, sampling stays conservative and tends to use the initial seed rather than fully normal search behavior.

### 4. Main async loop starts

The runner starts at least two ongoing loops:
- a proposal-generation coordinator
- a job monitor / completion processor

These run concurrently, so proposal generation and evaluation overlap.

### 5. A new proposal begins

For each new proposal:
1. Shinka reads the current meta summary / recommendations if enabled.
2. The LLM model sampler chooses which model to use for this proposal.
3. The database samples an island.
4. The database samples a parent program from that island.
5. The database samples archive inspirations and top-k inspirations.

If there are no correct programs available, Shinka switches into fix mode and samples an incorrect program plus its ancestry instead.

### 6. Prompt construction

Shinka builds the patch prompt:
- choose patch type: `diff`, `full`, or `cross`
- include the parent code
- include performance information
- include inspirations
- include optional text feedback
- include one sampled meta recommendation if available

If prompt evolution is enabled, it also samples which system prompt variant to use.

### 7. Patch generation

The chosen LLM produces a patch or rewritten program.

Shinka then:
1. extracts patch metadata like name and description
2. applies the patch
3. checks whether the patch actually changed the code
4. retries if patch application fails and attempts remain

Fix mode always uses a full rewrite. Normal mode uses the sampled patch type.

### 8. Embedding and novelty filtering

After a successful patch:
1. Shinka computes an embedding for the new code
2. if novelty checking is enabled, it compares the new code against programs in the same island
3. if the code is too similar, it may reject the proposal and retry from a fresh parent/inspiration sample
4. an optional LLM novelty judge can overrule pure embedding similarity

This is one of the main anti-collapse mechanisms.

### 9. Evaluation job submission

Once a proposal is accepted:
1. Shinka writes the generated code into that generation directory
2. submits the job to the configured scheduler
3. tracks the running job and its metadata

At this point the proposal becomes an evaluation job.

### 10. Job execution

The evaluator runs externally through the scheduler and writes results back:
- correctness
- public metrics
- private metrics
- combined score
- optional text feedback
- stdout / stderr logs

### 11. Completed job processing

When the job finishes, Shinka:
1. reads the result files
2. reconstructs a `Program` object
3. attaches lineage metadata, costs, embeddings, and patch details
4. inserts the program into the database

### 12. Database updates after insertion

Adding a program triggers several updates:
- assign island
- increment parent child count
- update archive if the program is correct
- update best-program tracking
- recompute embedding projections / clusters
- schedule migration if needed
- check stagnation and possibly spawn a new island

This is where most of Shinka's diversity and refinement logic actually lives.

### 13. Meta summarization

If meta summarization is enabled, every evaluated program is pushed into the meta memory buffer.

Once enough new programs accumulate:
1. summarize individual programs
2. synthesize global insights
3. generate actionable recommendations

Later proposals can sample one recommendation and inject it into the system prompt.

This is reflective guidance, but it is still grounded in already-tested programs.

### 14. Prompt evolution

If prompt evolution is enabled:
1. the system tracks which prompt generated each program
2. prompt fitness is updated from program outcomes
3. periodically a new system prompt is evolved from a parent prompt
4. future proposals can sample from the prompt archive

This creates a second adaptive loop around the program-evolution loop.

### 15. Bandit update

After a completed program is processed:
1. Shinka looks up which LLM model generated it
2. computes the reward from the resulting program performance
3. updates the LLM bandit
4. updates the observed cost for that model

That updated posterior affects future model selection.

### 16. Repeat until stop condition

Shinka keeps cycling through proposal generation, evaluation, and database updates until one of these happens:
- target number of generations reached
- API budget reached
- the system is explicitly stopped

Because the runner is async, generations can complete out of order. Completion is counted based on total finished work, not strict generation ordering.

### 17. Finalization

At shutdown Shinka:
- recomputes final embedding projections if possible
- runs a final meta summary if needed
- saves bandit state
- leaves a complete results directory with database, generated programs, logs, and meta outputs

## Practical Takeaway

Shinka explores in three different ways at once:
- **model exploration** via the LLM bandit
- **solution exploration** via islands, novelty checks, archive diversity, and crossover
- **prompt exploration** via prompt evolution

But the whole system is still centered on executable programs and measured outcomes. It is not a first-class conceptual search system.
