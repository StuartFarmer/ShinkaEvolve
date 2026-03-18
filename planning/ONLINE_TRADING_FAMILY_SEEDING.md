# Online Trading Family Seeding Spec

## Problem framing

This document defines the first family-seeding pass for an online trading task with:

- `X.shape == (200000, 100)`
- `r1.shape == (200000,)`
- at time `t`, the strategy observes `x_t = X[t]`
- the strategy emits `p_t in [-1, 1]`
- realized pnl is approximately `p_t * r1[t + 1]`
- updates are strictly online with no future leakage

This family catalog is intended for the future hierarchical layer. Each family should have:

- a stable `family_id`
- a reusable family context / hypothesis
- two initial runnable seed programs

The goal is not to find the best architecture up front. The goal is to create meaningfully different starting families that Shinka can test and refine.

## Common seed-program contract

All seed programs for this task should share one online interface so the evaluator and hierarchical runner can treat families uniformly.

Recommended contract:

```python
def run_online_strategy(
    X: np.ndarray,
    r1: np.ndarray,
) -> dict[str, object]:
    ...
```

Expected internal flow:

1. maintain online state
2. observe `x_t`
3. compute signal / score `s_t`
4. map `s_t` to bounded position `p_t`
5. realize reward from `r1[t + 1]`
6. update state using only information available after that reward

Expected outputs:

- `positions`
- `strategy_returns`
- `combined_score`
- diagnostics such as turnover, mean absolute position, and per-family metadata

## Shared implementation defaults

Use these defaults across all initial seed programs unless a family requires something else:

- features are standardized online with EW mean / variance normalization
- positions are bounded with `tanh`
- missing or non-finite features produce a neutral fallback contribution
- no transaction costs in v1 unless the evaluator already includes them
- every family records enough metadata to reconstruct its internal state evolution

Common bounded map:

```python
p_t = np.tanh(s_t / tau_t)
```

where `tau_t` can be constant or adaptive.

## Family catalog

### F1. `linear_rls`

Family context:
- predict next return directly with a linear model
- prefer simple, fast, heavily regularized online learners
- rely on forgetting to handle regime changes

Seed A:
- EW-standardized online ridge regression updated with SGD
- linear forecast of next return
- position from `tanh(pred / tau)`

Seed B:
- recursive least squares with forgetting factor
- diagonal stabilization / ridge floor
- position from `tanh(pred / ewma_vol)`

### F2. `logistic_directional`

Family context:
- predict direction first, then convert directional edge into exposure
- favor stable bounded decisions over precise magnitude estimation

Seed A:
- online logistic regression for `P(r_{t+1} > 0 | x_t)`
- position `2q_t - 1`

Seed B:
- online logistic sign model plus separate EW conditional magnitude model
- position `(2q_t - 1) * clipped_magnitude`

### F3. `mean_variance_control`

Family context:
- estimate expected return and risk separately
- size positions as certainty-equivalent exposure rather than pure forecast score

Seed A:
- online linear mean forecast
- EWMA volatility estimate
- position `clip(mu / (lambda * sigma^2 + eps), -1, 1)`

Seed B:
- online linear mean forecast
- downside-risk or drawdown-aware denominator
- stronger stabilization for low-vol blowups

### F4. `contextual_bandit`

Family context:
- treat each step as context-conditioned action selection
- learn action choice directly rather than pure return forecasting

Seed A:
- LinUCB over actions `{-1, 0, 1}`
- reward is realized pnl

Seed B:
- linear Thompson sampling over actions `{-1, -0.5, 0, 0.5, 1}`
- Bayesian action uncertainty drives exploration

### F5. `policy_gradient_rl`

Family context:
- parameterize the policy directly and optimize pnl-like reward
- allow objective-level control over turnover and risk

Seed A:
- linear tanh policy
- simple online REINFORCE-style update
- reward includes optional turnover penalty hook

Seed B:
- one-hidden-layer tanh policy
- actor-only update with strong gradient clipping and reward normalization

### F6. `state_space`

Family context:
- use latent state or rolling memory to make the same feature mean different things in different regimes
- favor lightweight statefulness before full neural recurrence

Seed A:
- exponential latent state over standardized features
- linear readout from latent state to position

Seed B:
- compact GRU-style recurrent model with tiny hidden dimension
- aggressive regularization and bounded hidden state updates

### F7. `mixture_of_experts`

Family context:
- combine several specialized online models
- gating should decide which expert matters now

Seed A:
- 5 simple experts:
  - trend
  - mean reversion
  - breakout
  - low-vol
  - linear baseline
- multiplicative-weights expert allocator

Seed B:
- same expert bank
- softmax gate driven by recent context and expert performance

### F8. `online_trees_boosting`

Family context:
- capture nonlinear interactions and threshold effects that linear models miss
- keep the first seeds shallow and regularized

Seed A:
- shallow rolling-window tree regressor retrained periodically
- position from bounded forecast

Seed B:
- shallow boosted stump ensemble with strict depth / update caps
- heavy shrinkage to avoid instability

### F9. `kernel_similarity`

Family context:
- use local analogs in feature space rather than global parametric structure
- restrict memory so the method remains online and affordable

Seed A:
- rolling-buffer online kNN regressor on standardized features
- recency-weighted neighbor averaging

Seed B:
- random Fourier features plus online linear regression
- acts like a cheap approximate kernel model

### F10. `alpha_meta_weighting`

Family context:
- treat each of the 100 columns as an alpha sleeve or micro-signal
- learn dynamic weights over sleeves instead of one monolithic predictor

Seed A:
- normalize each feature into a sleeve signal
- combine with Hedge / exponential weights

Seed B:
- grouped multiplicative-weights allocator with shrinkage
- include correlation damping or weight clipping

## Priority order for first testing pass

Implement and test these families first:

Tier 1:
- `alpha_meta_weighting`
- `mixture_of_experts`
- `mean_variance_control`
- `linear_rls`

Tier 2:
- `logistic_directional`
- `contextual_bandit`
- `kernel_similarity`

Tier 3:
- `state_space`
- `policy_gradient_rl`
- `online_trees_boosting`

This priority order matches the practical tradeoff between likely usefulness, debuggability, and implementation fragility.

## Seeding policy for the hierarchical runner

For this task, the hierarchical layer should seed:

- 10 total families
- 2 seed programs per family
- 20 initial seed programs total

Directory convention for future implementation:

```text
results/<run>/families/<family_id>/
    family_context.md
    seed_a.py
    seed_b.py
```

Each `family_context.md` should contain:

- family hypothesis
- expected edge
- update rule summary
- risk sizing rule
- likely failure modes
- reasons this family is distinct from the others

## Strong practical interpretation

For this specific problem, the most important split is:

- **direct forecast families**:
  - `linear_rls`
  - `logistic_directional`
  - `mean_variance_control`
  - `state_space`
  - `online_trees_boosting`
  - `kernel_similarity`

- **alpha aggregation / action selection families**:
  - `contextual_bandit`
  - `policy_gradient_rl`
  - `mixture_of_experts`
  - `alpha_meta_weighting`

If the 100 columns are already alpha-like signals, the second group should get priority.

## Recommended next implementation step

The next concrete build step should be:

1. define the evaluator contract for `X` and `r1`
2. implement the common online seed-program API
3. create family context files for the 10 families
4. implement the 8 Tier-1 / Tier-2 seeds first
5. leave Tier-3 seeds as optional second-pass prototypes

This keeps the first family-seeding pass broad enough to test the search hierarchy without spending too much budget on the most fragile model classes.
