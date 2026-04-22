# 0002: `StateEstimator.fit` takes a cohort mapping; `infer` takes a single `Channels`

**Date:** 2026-04-22
**Status:** accepted

## Decision

`StateEstimator.fit` takes `cohort: Mapping[str, Channels]` and learns one shared parameter set across all athletes. `StateEstimator.infer` takes a single `channels: Channels` argument (plus an optional `prior: Prior | None` override) and returns that athlete's `Z` trajectory under the frozen parameters — for any athlete, seen or unseen at fit time. The fitted estimator holds parameters only; no per-athlete data is cached.

## Context

`docs/decisions/0001-shared-parameters-per-athlete-Z.md` committed the project to shared parameters across athletes with between-athlete variation absorbed into per-athlete `Z`. But the scaffold's signatures (`architecture_map.md` §3.4 and §3.7) accepted only one `Channels`:

```python
# scaffolded
def fit(self, channels: Channels, ...) -> "StateEstimator": ...
def infer(self, mode: InferMode = "filter") -> Z: ...
```

That made ADR 0001 unrealizable as written — one `fit` call couldn't see a cohort. An intermediate proposal (`fit` takes a cohort, `infer(subject_id)` looks up a seen athlete) solved the shared-fit problem but couldn't handle validation-cohort athletes whose `subject_id` was never in the fit call. This ADR resolves both halves in one move.

## Alternatives considered

- **Make `Channels` multi-athlete.** `subject_id` becomes an `Array`, and every `(T, d)` array becomes `(sum_T, d)` with boundaries tracked. Rejected: invasive change to every downstream module, and conflates typing (one athlete's DAG primitives) with fitting (plurality of athletes). Violates CLAUDE.md §3 (surgical changes) without a corresponding gain.
- **Two `infer` methods — `infer(subject_id)` for seen athletes, `infer_new(channels)` for unseen.** Rejected: duplicates surface area for what is the same operation under ADR 0001 (run inference on this athlete's `Channels` under frozen parameters). The seen/unseen distinction is a harness-level concern, not an estimator concern.
- **Unified `infer(subject_id, channels=None)`.** Rejected: the `None` path implicitly requires `fit` to cache per-athlete data, which mixes parametric fitting with per-athlete bookkeeping. Hard to reason about and harder to test.

## DAG implications

- Affects the `StateEstimator` Protocol in `statepace/filter.py`; no DAG edge moves.
- Relies on A7 (between-athlete variation lives in `Z`) in its strongest reading — since every athlete's `Z` is produced by the same parametric estimator, the per-athlete state is the only carrier of between-athlete difference.
- Relies on A10 (stationarity of shared parameters across athletes) — a natural extension of A10's within-athlete statement that the shared-parameter fit is already committed to by ADR 0001.
- Does not affect the factorization `p(X_t | Z_{t-1}, P_t, E_t) · p(Z_t | Z_{t-1}, X_t) · p(Z_t | Z_{t-1})`.

## Empirical grounding

None. Architectural decision taken before any estimator is implemented. The first empirical check on whether the shared-parameter / per-athlete-`Z` story holds up will be the test-vs-validation-score gap reported in a future ADR after the first estimator runs.

## What would invalidate this

- Per-athlete re-inference cost becomes prohibitive at scale (the accepted tradeoff is that training-cohort inference re-runs against the same `Channels` that `fit` already saw). Mitigations — explicit caching layer, memoized estimator wrapper — would be localized and not require revisiting the Protocol.
- A future estimator family for which the per-athlete inference is *not* a pure function of `(parameters, channels)` — e.g., requires a per-athlete fitted quantity that can't be recomputed from `channels` alone. Would force reintroducing per-athlete state on the estimator object.
- Per-athlete informative priors on `Z_0` (the flagged A8 concern) are already accommodated by the optional `prior` argument on `infer`, so that extension is not an invalidation.

## Follow-ups

- Update `architecture_map.md` §3.4 (`StateEstimator` Protocol), §3.7 (`EvalSplit` gains a `cohort` label; `make_splits`, `run_evaluation`, `EvalResult` take/return keyed mappings), and add a sibling boundary rule in §4 that estimator pooling is invisible to observation and transitions — they continue to see per-athlete arrays only.
- Update the scaffold files (`statepace/filter.py`, `statepace/evaluation/harness.py`) to match the new signatures. Bodies remain `...`.
- The A8 revision for trait-driven `Z_0` initialization is a separate future ADR; the optional `prior` argument on `infer` is the extension hook.
- **Within-cohort weak identification.** An identifiability audit (2026-04-22) surfaced a structural concern: per-athlete `Z` has `O(N × T × d_Z)` degrees of freedom against `O(1)` shared parameters, so per-athlete `Z` trajectories may absorb variation that shared parameters should explain. The test-vs-validation split detects across-cohort generalization failure but not within-cohort weak identification (two parameter settings can give equal training-cohort likelihood because per-athlete `Z` re-absorbs the difference). The ADR that selects the first `StateEstimator` family must name a concrete within-cohort diagnostic — e.g., fit on random cohort halves and compare inferred `Z` trajectories on shared athletes. No Protocol additions are needed for that diagnostic; it uses only `fit`, `infer`, and `EvalResult`.
