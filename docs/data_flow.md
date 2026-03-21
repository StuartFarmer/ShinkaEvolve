# Shinka Data Flow

This document shows the current end-to-end data flow in Shinka as it exists
today. It is intentionally descriptive, not aspirational.

The diagrams below reflect the current architecture:

- runner-centric orchestration
- SQLite-backed `ProgramDatabase`
- prompt sampling + patch generation
- optional novelty gating
- evaluation
- re-ingestion into the database

## 1. Full Run Flow

```mermaid
flowchart TD
    A[CLI / Python entrypoint] --> B[Load config objects]
    B --> C[Create ShinkaEvolveRunner]
    C --> D[Init LLM clients]
    C --> E[Init ProgramDatabase]
    C --> F[Init PromptSampler]
    C --> G[Init JobScheduler]
    C --> H[Init MetaSummarizer optional]
    C --> I[Init NoveltyJudge optional]
    C --> J[Init Prompt Evolution optional]

    J --> K[Async generation loop]
    I --> K
    H --> K
    G --> K
    F --> K
    E --> K
    D --> K

    K --> L[DB sample_with_fix_mode]
    L --> M[parent Program]
    L --> N[archive inspirations]
    L --> O[top-k inspirations]
    L --> P[needs_fix flag]

    M --> Q[PromptSampler.sample or sample_fix]
    N --> Q
    O --> Q
    P --> Q
    H --> Q

    Q --> R[system prompt + user prompt + patch type]
    R --> S[LLM proposal call]
    S --> T[Patch/full/cross application]
    T --> U[candidate code written to exec file]

    U --> V[Embedding client]
    V --> W[candidate code embedding]

    W --> X{Run novelty gate?}
    X -->|no| Y[Submit candidate to evaluator]
    X -->|yes| Z[Novelty judge]

    Z --> AA{Accept?}
    AA -->|no| AB[Resample parent/inspirations/patch]
    AB --> L
    AA -->|yes| Y

    Y --> AC[JobScheduler executes evaluate.py]
    AC --> AD[metrics.json / correct.json / result artifacts]
    AD --> AE[Runner reads evaluation results]
    AE --> AF[Construct Program object]
    AF --> AG[ProgramDatabase.add]

    AG --> AH[assign island]
    AG --> AI[store metrics + metadata + feedback]
    AG --> AJ[update children_count]
    AG --> AK[update best program id]
    AG --> AL[archive maintenance]
    AG --> AM[migration / island bookkeeping]

    AM --> K
    AL --> K
    AK --> K
```

## 2. Proposal Loop in More Detail

```mermaid
flowchart LR
    A[Target generation g] --> B[Sample parent context]
    B --> C[Build proposal prompt]
    C --> D[Call proposal LLM]
    D --> E[Patch application / code extraction]
    E --> F[Candidate code file]
    F --> G[Compute code embedding]
    G --> H{Novel enough?}
    H -->|No| I[Reject candidate]
    I --> B
    H -->|Yes| J[Evaluate candidate]
    J --> K[Build Program]
    K --> L[Store in DB]
```

## 3. What `ProgramDatabase.sample(...)` Is Actually Doing

```mermaid
flowchart TD
    A[ProgramDatabase.sample] --> B{All islands initialized?}
    B -->|No| C[Return initial program only]
    B -->|Yes| D[IslandSampler.sample_island]
    D --> E[CombinedParentSelector.sample_parent]
    E --> F[parent Program]
    F --> G[CombinedContextSelector.sample_context]
    G --> H[ArchiveInspirationSelector]
    G --> I[TopKInspirationSelector]
    H --> J[archive inspiration Programs]
    I --> K[top-k inspiration Programs]
    F --> L[Return sampled context]
    J --> L
    K --> L
```

## 4. Prompt Assembly Flow

```mermaid
flowchart TD
    A[parent Program] --> B[PromptSampler]
    C[archive inspirations] --> B
    D[top-k inspirations] --> B
    E[meta recommendations optional] --> B
    F[parent text_feedback optional] --> B

    B --> G[Choose patch type]
    G --> H[diff]
    G --> I[full]
    G --> J[cross]

    C --> K[InspirationContextBuilder]
    D --> K
    K --> L[sorted inspiration history block]

    H --> M[compose system prompt]
    I --> M
    J --> M
    L --> N[compose user prompt]
    F --> N

    M --> O[LLM input]
    N --> O
```

## 5. Novelty Path

```mermaid
flowchart TD
    A[candidate code file] --> B[read code]
    C[candidate embedding] --> D[compute_similarity against island]
    E[parent Program] --> D

    D --> F[similarity scores list]
    F --> G[max_similarity]
    G --> H{max_similarity <= threshold?}

    H -->|Yes| I[Accept candidate]
    H -->|No| J[get_most_similar_program]
    J --> K[nearest Program]
    A --> L[check_llm_novelty]
    K --> L
    L --> M{LLM says NOVEL?}
    M -->|Yes| I
    M -->|No| N[Reject candidate]
```

Important current behavior:

- similarity is computed against all embedded programs in the parent's island
- only the single nearest neighbor is passed to the novelty LLM
- the novelty metadata is mostly stored for observation, not reused later for
  algorithmic control

## 6. Evaluation Feedback Loop

```mermaid
flowchart TD
    A[candidate code] --> B[evaluate.py]
    B --> C[combined_score]
    B --> D[public_metrics]
    B --> E[private_metrics]
    B --> F[text_feedback]
    B --> G[artifact files]

    C --> H[Program object]
    D --> H
    E --> H
    F --> H

    H --> I[ProgramDatabase.add]
    F --> J[future PromptSampler prompt]
    I --> J
```

This is the main evaluator-to-proposer feedback path:

- evaluator writes `text_feedback`
- runner stores it on `Program`
- `PromptSampler` injects parent `text_feedback` into future prompts if
  `use_text_feedback=true`

## 7. Current Data Ownership

```mermaid
classDiagram
    class ShinkaEvolveRunner {
        +run()
        +proposal loop
        +novelty orchestration
        +evaluation orchestration
        +meta summarization
        +prompt evolution
    }

    class ProgramDatabase {
        +add(program)
        +sample()
        +sample_with_fix_mode()
        +compute_similarity()
        +get_most_similar_program()
        +archive maintenance
        +best program tracking
        +migration bookkeeping
    }

    class PromptSampler {
        +sample()
        +sample_fix()
    }

    class NoveltyJudge {
        +should_check_novelty()
        +assess_novelty_with_rejection_sampling()
        +check_llm_novelty()
    }

    class JobScheduler {
        +submit_async_nonblocking()
        +read results
    }

    class Program {
        +code
        +embedding
        +metrics
        +text_feedback
        +metadata
    }

    ShinkaEvolveRunner --> ProgramDatabase
    ShinkaEvolveRunner --> PromptSampler
    ShinkaEvolveRunner --> NoveltyJudge
    ShinkaEvolveRunner --> JobScheduler
    ProgramDatabase --> Program
```

## 8. Where the Coupling Is

```mermaid
flowchart TD
    A[Runner] --> B[DB sample logic]
    A --> C[Prompt logic]
    A --> D[Novelty logic]
    A --> E[Evaluation logic]
    A --> F[Program ingestion]

    B --> G[parent selection]
    B --> H[inspiration selection]
    B --> I[island management]
    B --> J[archive management]
    B --> K[embedding similarity]

    D --> K
    F --> J
    F --> I
    E --> L[text feedback]
    L --> C
```

This is the main reason the system feels spaghetti-like:

- `ProgramDatabase` is both persistence and search policy
- `Runner` is both orchestrator and lifecycle owner of almost everything
- novelty and retry behavior are split between judge and runner
- prompt policy and scheduling are not clearly separated

## 9. Rough Target Refactor

This is the cleaner shape to move toward:

```mermaid
flowchart TD
    A[Manual / high-level controller] --> B[ProgramRepository]
    A --> C[ContextSampler]
    A --> D[PromptBuilder]
    A --> E[ProposalGenerator]
    A --> F[NoveltyContextProvider]
    A --> G[NoveltyJudge]
    A --> H[Evaluator]
    A --> I[ProgramFactory / Ingestor]

    B --> J[(SQLite / ORM)]
    F --> J
    I --> J
    C --> B
    C --> K[ParentSelector]
    C --> L[InspirationSelector]
    C --> M[IslandService]
```

That would separate:

- persistence
- search policy
- proposal generation
- novelty decision
- orchestration

instead of forcing all of them through `ShinkaEvolveRunner` and
`ProgramDatabase`.
