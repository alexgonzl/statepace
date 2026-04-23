# Post-Scaffold Sequencing

**Status:** active
**Last updated:** 2026-04-22 (D2/D3 resolved; M2 implementation partially landed)

The sequencing plan for work after the modeling-pipeline scaffold. Produces the project's two outputs — state estimation (`Z` trajectory) and next-workout prediction (`X̂`) — under a proper train / test / validation split.

---

## Current state

### Completed

| # | Milestone | Commits | Notes |
|---|---|---|---|
| M0 | ADR 0001 — shared params, per-athlete Z | `2d432d5` | Seeded `docs/decisions/`. |
| — | ADR 0002 — cohort-in-fit, channels-in-infer | `b8fa1a9` | Emerged from the architect's closing pass on the multi-athlete container question; not pre-enumerated in the plan. |
| M1a | Synthetic fixture factory + tests | `9a5e9c2` | `tests/fixtures/synthetic.py` + `tests/test_channels.py`. 6/6 new tests pass; 8/8 suite-wide. |
| — | ADR 0003 — cohort-assignment (stratified by sex + volume) | `ce5fd11` | Resolves D3. Introduces `AthleteMeta` sibling to `Channels`; architecture_map §3.1/§3.7/§4 updated (new boundary rule 13). |
| M2 (partial) | `AthleteMeta` dataclass + synthetic fixture factory | `19bc32e` | `statepace/channels.py` + `tests/fixtures/synthetic.py::make_athlete_meta`. 8/8 suite-wide. `assign_cohorts` / `make_splits` impl still pending. |

### Active

**M2 — Split / cohort machinery.** D2 and D3 resolved. `AthleteMeta` landed; remaining work is the `assign_cohorts` / `make_splits` implementation. Blocks M8, M9, M10.

### Audits landed

- **Identifiability-auditor on ADR 0002** (2026-04-22): surfaced a structural pooling-vs-local-latent tension — `O(N·T·d_Z)` per-athlete state degrees of freedom vs `O(1)` shared parameters. Proposed three Protocol additions to the `StateEstimator`.
- **Senior-scientist second opinion** (2026-04-22): rejected the auditor's `score(channels)` as functional-form contamination; deferred the parameters handle; accepted `d_Z` as a class attribute. Within-cohort weak-identification diagnostic recommendation captured in ADR 0002's follow-ups.
- **Master-architect on `AthleteMeta` placement** (2026-04-22): placed alongside `Channels` in `statepace/channels.py` as a sibling dataclass (not a new module); flagged the session-vs-athlete boundary as boundary rule 13; deferred `__init__.py` export to the M2 implementation commit. Design fed into ADR 0003.

---

## Constraints (load-bearing)

- **Two outputs:** state estimation (`Z` trajectory) and next-workout prediction (`X̂`).
- **Split structure:**
  - Athletes partitioned into `training_cohort` and `validation_cohort` (disjoint by `subject_id`).
  - Each athlete's timeline is `[ warmup | train | test ]`.
  - Training cohort: one model fit across all training athletes' train portions (post-warm-up). Scoring on their test portions is the **test** score.
  - Validation cohort: shared parameters frozen; each validation athlete's warmup + train used only to estimate their `Z` trajectory under the fitted parameters. Scoring on their test portion is the **validation** score.
- **Shared parameters, per-athlete `Z`** (ADR 0001, §A7 framing).
- **`StateEstimator.fit` takes `Mapping[str, Channels]`; `infer` takes a single `Channels`** (ADR 0002). Same code path for seen and unseen athletes.
- **No functional-form assumptions** for the observation model, `f`, or `g`. Only architecture-level vocabulary appears here.
- **No hardcoded values.** Every length, threshold, or count is a hyperparameter surfaced by name.

---

## Milestones (dependency-ordered)

```
M0  ADR 0001 — shared params, per-athlete Z       ✅ done (2d432d5)
M1a Synthetic fixtures                              ✅ done (9a5e9c2)
M1b Real-data fixture                               ⏸ gated on D5 (external)
M2  Split/cohort machinery                          ▶ active — gated on D2, D3
M3  evaluation/deconfounding.py                    ⏸ ready; parallel with M2
M4  ObservationModel (one concrete impl)           ⏸ gated on D6
M5  WorkoutTransition + RestTransition             ⏸ gated on D7
M6  StateEstimator (one concrete impl)             ⏸ gated on D8 + M4 + M5
M7  forward.py + predict.py (glue)                 ⏸ gated on M5
M8  evaluation/metrics.py + harness wiring         ⏸ gated on M4–M7 + D10
M9  End-to-end run on synthetic                    ⏸ gated on M8
M10 End-to-end run on real data                    ⏸ gated on M9 + M1b
M_verdict ADR — verdict on the first estimator    ⏸ gated on M10
```

Parallelizable now: `{M2, M3}`. Everything post-M3 serial on the dependency chain.

---

## Milestone detail

### M1b — Real-data fixture

Wire one real-athlete subset to confirm `docs/data_contract.md` matches actual data.

**Decision gate:** D5 (real-data availability + path).

**Critical files:** location TBD; expected under `tests/fixtures/` following the directory-governance rule.

### M2 — Split / cohort machinery

Implement the train/test/validation split per ADR 0001 + ADR 0002 + ADR 0003. Lives in `statepace/evaluation/harness.py` (signature updated in architecture_map §3.7 to `make_splits(cohort: Mapping[str, Channels], meta: Mapping[str, AthleteMeta], warmup_days: int) -> Iterable[EvalSplit]`).

**Work:**
- `assign_cohorts(athletes, meta, *, validation_fraction, seed, volume_bucket_edges)` — stratified random split on (`sex`, volume bucket); per ADR 0003.
- `make_splits` produces per-athlete `EvalSplit` objects with `cohort: Literal["train", "test", "validation"]`.
- For training-cohort athletes: `fit_idx = post-warmup train indices`, `score_idx = test indices`.
- For validation-cohort athletes: same structure; `cohort="validation"` so scoring reports separately.
- Warm-up mask enforcement at harness edge (boundary rule 6).

**Landed (as of `19bc32e`):** `AthleteMeta` dataclass in `statepace/channels.py`; `make_athlete_meta` in `tests/fixtures/synthetic.py`; shape-parity tests.

**Remaining:** `assign_cohorts` + `make_splits` implementation + M2 verification tests (using `warmup_days=90, train_days=210, test_days=60` per D2).

**Hyperparameters surfaced by name (no defaults):** `warmup_days`, `train_days`, `test_days`, `validation_fraction`, `seed`, `volume_bucket_edges`.

**Decision gates:** D2 ✅ (`90 / 210 / 60`), D3 ✅ (ADR 0003).

**Critical files:** `statepace/evaluation/harness.py`; possibly `statepace/evaluation/splits.py` (architect call when implementation lands).

### M3 — `evaluation/deconfounding.py`

Reference-template projection per `docs/conventions.md` and architecture_map §3.8. Pure scoring convention. Reference template (`sea level, flat, 5k, noon, 12°C wet-bulb`) is defined in conventions — pulled at evaluation time, not redefined.

**Critical files:** `statepace/evaluation/deconfounding.py` (new).

### M4 — `ObservationModel` implementation

**Gated on D6.** Once the functional form of the observation model is chosen (its own ADR), implement one concrete class conforming to the scaffold Protocol.

**Critical files:** `statepace/observation.py` (replace scaffold bodies).

### M5 — `WorkoutTransition` + `RestTransition` implementations

**Gated on D7.** Transitions own the rest-day bound (`max_consecutive_rest_days`).

**Critical files:** `statepace/transitions.py` (replace scaffold bodies).

### M6 — `StateEstimator` implementation

**Gated on D8 + M4 + M5.** Must respect `Prior.diffuse=True` (A8), warm-up masking at harness (boundary rule 6), emit `Z` over full history. The ADR selecting the family must name a within-cohort weak-identification diagnostic (per ADR 0002's follow-ups).

**Critical files:** `statepace/filter.py` (replace scaffold bodies).

### M7 — `forward.py` + `predict.py`

Glue. `forward_state` propagates `Z` through a `ForwardSchedule`. `predict_session` composes `filter → forward → observation.inverse` at queried `(P, E)`.

**Critical files:** `statepace/forward.py`, `statepace/predict.py` (replace scaffold bodies).

### M8 — `evaluation/metrics.py` + harness wiring

- State metric(s) on `Z_hat` — trajectory diagnostics.
- Prediction metric(s) on `X_pred` — residual-based; specific form is D10.
- `run_evaluation` wires everything end-to-end; returns `EvalResult` keyed by `subject_id` with cohort labels.

**Hyperparameters surfaced by name:** τ-horizons to evaluate; metric choice.

**Critical files:** `statepace/evaluation/metrics.py` (new), `statepace/evaluation/harness.py`.

### M9 — End-to-end run on synthetic

Pipeline runs over the synthetic cohort from M1a. Verifies composition and shape alignment. Does not claim parameter recovery (M1a fixtures have no ground-truth).

### M10 — End-to-end run on real data

Pipeline runs over the real cohort from M1b. First actual test and validation scores.

### M_verdict — ADR on the first estimator family

After M10, record what was recovered, what the test-vs-validation gap looks like, and conditions for revisiting the estimator choice. Numbering assigned at time of writing (next unused ADR number).

---

## Open decisions

| # | Decision | Needed before | Status |
|---|---|---|---|
| D1 | Shared params, per-athlete `Z` | M0 | ✅ taken — ADR 0001 |
| D2 | `warmup_days`, `train_days`, `test_days` values for M2 tests | M2 | ✅ resolved — `90 / 210 / 60` (T=360) |
| D3 | Cohort-assignment procedure | M2 | ✅ taken — ADR 0003 |
| D4 | Synthetic cohort sizing for tests (`n_athletes`, `n_days`) | M1a | ✅ resolved inline — factories parameterized, test files decide |
| D5 | Real-data availability + path | M1b | ⏸ open (external) |
| D6 | Functional form of `p(X | Z, P, E)` | M4 | ⏸ out of plan; separate scoping session |
| D7 | Functional forms of `f` and `g` | M5 | ⏸ out of plan; separate scoping session |
| D8 | `StateEstimator` family | M6 | ⏸ out of plan; separate scoping session |
| D9 | Dimensionality of `Z` (`d_Z`) | M4 (load-bearing downstream) | ⏸ open |
| D10 | Prediction metric(s) and τ-horizons | M8 | ⏸ open — own ADR |

---

## Verification per milestone

| Milestone | Verification |
|---|---|
| M0 | ADR 0001 exists; `docs/decisions/` created; CLAUDE.md pointer resolves |
| M1a | `pytest tests/test_channels.py` — synthetic fixtures match §3.1 schema |
| M1b | Real-data fixture loads into `Channels` without contract violation |
| M2 | Unit test: `make_splits` produces correct per-cohort `EvalSplit` objects; warm-up masked from `score_idx`; no date appears in both `fit_idx` and `score_idx` of a given athlete |
| M3 | Unit test: reference-template projection roundtrip; idempotent at the template |
| M4 | Unit test against Protocol: `forward` returns `X` of correct shape; `inverse` recovers a held-out component |
| M5 | Unit test against Protocol: `step` returns `Z` of correct shape; rest-bound contract enforced |
| M6 | Unit test against Protocol: `fit` + `infer(channels, mode="filter")` emits `Z` over full history; warm-up indices not masked at estimator level (harness's job) |
| M7 | Unit test: `forward_state` propagates τ days through a `ForwardSchedule`; `predict_session` respects `ConditioningSpec` |
| M8 | Unit test: metric functions return arrays of expected shape; cohort-separated scoring |
| M9 | Integration: pipeline runs end-to-end on synthetic; both cohorts produce `EvalResult` |
| M10 | Integration: real-data cohorts produce numeric test and validation scores; warm-up and rest-bound flags report expected counts |
| M_verdict | ADR exists and is reviewed |

---

## Out of scope (explicit — will not creep back in)

- Functional forms of the observation model, `f`, `g`. Each is its own ADR-scoped decision.
- Estimator family choice. Own ADR.
- Selection model `p(P|Z)`. Deferred per architecture_map §5.1.
- Re-entry policy past the rest bound. Deferred per architecture_map §5 item 2.
- Health-state projection `h(Z_t)`. Not on the prediction path.
- Counterfactual / marginalized-`P` prediction modes. Blocked on selection model.
- CI/dev pipelines.
- Cross-athlete parameter sharing beyond ADR 0001's shared-params assumption (e.g., hierarchical priors).

---

## Maintenance

This plan is the canonical current state of in-flight work. Update rules (enforced as a PM obligation):

- **At each milestone completion:** move the milestone from Active → Completed, record the commit SHA.
- **At each audit:** add a line under "Audits landed" with date and one-sentence finding.
- **When a decision resolves:** flip the status in the Open decisions table.
- **When scope drifts:** update Out of scope explicitly before the drift lands.
- **Plan edits are their own commits** — small, named, and tied to the milestone they record.

Retired plans stay in `docs/plans/` with `**Status:** complete` or `**Status:** superseded by …`.
