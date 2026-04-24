# Post-Scaffold Sequencing

**Status:** active
**Last updated:** 2026-04-23 (Track B: `riegel-score-hrstep` spec drafted)

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
| M2a | `AthleteMeta` dataclass + synthetic fixture factory | `19bc32e` | `statepace/channels.py` + `tests/fixtures/synthetic.py::make_athlete_meta`. 8/8 suite-wide. |
| M2b | `assign_cohorts` + `make_splits` + N=50 test cohort | `ee3dd76` | `statepace/evaluation/harness.py` + `tests/test_evaluation_harness.py`. 9 new tests; 19/19 suite-wide. architecture_map §3.7 `make_splits` signature extended with `train_days`, `test_days`, `cohort_assignment`. |
| — | Sweep-harness signatures (`FormsBundle`, `run_sweep`, `SweepResult`) | `cb81648`, `d7dc83e` | Signature-only landing in `statepace/evaluation/harness.py` and `docs/architecture_map.md` §3.7. Reframes D6/D7/D8 from "pick one form" to "sweep across forms." Body lands at M4 alongside the first reference `ObservationModel`. 19/19 suite-wide (unchanged). |
| — | ADR 0004 — prediction metric + τ-horizons (D10) | `ba6f832` | Two residual streams (raw + reference), five stats each, per-subject aggregation, three-way cohort reporting. Top-speed subset gated by per-athlete Riegel fit over `[0, warmup_days + train_days)`. Exports follow-ups to M4 (`make_splits` in-sample split) and M8 (`project_to_reference` body). |
| M4 Track A | `make_splits` in-sample train-slot | `3a47ca0` | Training-cohort athletes emit two `EvalSplit`s (`"train"` with `score_idx = [warmup_days, warmup_days + train_days)` and `"test"` with the post-train window); validation-cohort athletes emit one `"validation"` split. N=50 synthetic cohort (seed=0): 94 splits (44+44+6). 19/19 suite-wide. |
| — | ADR 0005 — `X_t` single-node + functionals | `5bf4e68` | Arbitrated outcome of Track B framework challenge: node-hood = independent intervenability (Pearl SCM); functionals `π_obs`/`π_stim` are per-family. §4(c) rewritten; A1 extended. Reference-impl specs must name both functionals. |
| — | `docs/reference_impls/` seeded | `2397796` | Governance README lands ahead of the first spec. |
| M4 Track B spec | `riegel-score-hrstep` (draft) | `66d28d8` | First reference `ObservationModel` spec. `π_obs` = best-effort-by-Riegel-relative-speed (5 components); `π_stim` = 5 whole-session load aggregates. Gaussian family, linear mean, `d_Z = 4`. Three rounds of expert audit recorded in-spec. Next: focused-engineer implementation. |

### Active

M4 Track B: first reference `ObservationModel` + `run_sweep` body. Track A landed (`3a47ca0`); Track B is the remaining M4 work and carries its own ADR.

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
M2  Split/cohort machinery                          ✅ done (ee3dd76)
M3  evaluation/deconfounding.py                    ⏸ deferred — concrete `ReferenceTemplate` needs named P/E components (CLAUDE.md: defer naming to M9/M10); scaffold-only landing optional
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

**Landed:**
- `19bc32e` — `AthleteMeta` dataclass in `statepace/channels.py`; `make_athlete_meta` in `tests/fixtures/synthetic.py`; shape-parity tests.
- `ee3dd76` — `assign_cohorts` and `make_splits` in `statepace/evaluation/harness.py`; N=50 synthetic test cohort (`make_m2_test_cohort`); 9 new verification tests. architecture_map §3.7 `make_splits` signature extended with keyword-only `train_days`, `test_days`, `cohort_assignment`. Volume summed over the train window (pre-test, no leakage into scoring).

**Hyperparameters surfaced by name (no defaults):** `warmup_days`, `train_days`, `test_days`, `validation_fraction`, `seed`, `volume_bucket_edges`, `volume_component`.

**Decision gates:** D2 ✅ (`90 / 210 / 60`), D3 ✅ (ADR 0003).

**Critical files:** `statepace/evaluation/harness.py`, `tests/test_evaluation_harness.py`, `tests/fixtures/synthetic.py`.

### M3 — `evaluation/deconfounding.py`

Reference-template projection per `docs/conventions.md` and architecture_map §3.8. Pure scoring convention. Reference template (`sea level, flat, 5k, noon, 12°C wet-bulb`) is defined in conventions — pulled at evaluation time, not redefined.

**Status:** deferred. Implementing `project_to_reference` concretely requires (i) a working `ObservationModel` (M4, gated on D6) and (ii) concrete P/E component names for the reference values. The late-naming principle (CLAUDE.md Standing Rules) defers (ii) to M9/M10. Scaffold-only landing (`ReferenceTemplate` dataclass + `project_to_reference` signature with `...` body) is possible now but adds no verifiable content until both dependencies resolve.

**Critical files:** `statepace/evaluation/deconfounding.py` (new).

### M4 — First reference `ObservationModel` + `run_sweep` body

Ship (a) one concrete `ObservationModel` conforming to the scaffold Protocol, and (b) the `run_sweep` body that iterates over `FormsBundle`s and populates a `SweepResult`. The reference implementation carries its own ADR declaring its family and compatibility posture (X-parameterization stance, Z-gauge choice — identifiability-baseline §3); the framework itself commits to no privileged form.

Additional reference implementations are additive and land in later milestones or separately from the sequencing chain. Each new reference impl ships with its own ADR.

Also lands at M4 per ADR 0004: `make_splits` emits an additional in-sample `EvalSplit` for each training-cohort athlete (`cohort="train"`, `score_idx` inside `[warmup_days, warmup_days + train_days)`) so the three-way cohort reporting has a train-slot to populate. Existing `make_splits` tests update in the same commit.

**Critical files:** `statepace/observation.py` (first reference impl), `statepace/evaluation/harness.py` (`run_sweep` body, duplicate-label check, `make_splits` in-sample split).

**Track A landed** (`3a47ca0`): `make_splits` in-sample split. Training-cohort athletes emit two splits (`"train"` + `"test"`); validation-cohort emit one (`"validation"`).

**Track B scoping surfaced a framework question** that landed as ADR 0005 (`5bf4e68`): single-`X_t` node with per-edge functionals `π_obs` / `π_stim`. Reference impl now specifies both functionals as part of its spec.

**Track B spec landed** (`66d28d8`): `docs/reference_impls/riegel-score-hrstep.md` in draft. Remaining Track B work: focused-engineer implementation of the reference `ObservationModel` against the spec, plus `run_sweep` body + duplicate-label check.

### M5 — First reference `WorkoutTransition` + `RestTransition`

Ship one concrete pair. Transitions own the rest-day bound (`max_consecutive_rest_days`). The ADR covering this reference pair declares its coherence posture against the M4 reference observation (dual-role `X`, per identifiability-baseline §3 concern 1).

**Critical files:** `statepace/transitions.py` (first reference impls).

### M6 — First reference `StateEstimator`

Gated on M4 + M5. Must respect `Prior.diffuse=True` (A8), warm-up masking at harness (boundary rule 6), emit `Z` over full history. The ADR covering this reference estimator must name a within-cohort weak-identification diagnostic (per ADR 0002's follow-ups).

**Critical files:** `statepace/filter.py` (first reference impl).

### M7 — `forward.py` + `predict.py`

Glue. `forward_state` propagates `Z` through a `ForwardSchedule`. `predict_session` composes `filter → forward → observation.inverse` at queried `(P, E)`.

**Critical files:** `statepace/forward.py`, `statepace/predict.py` (replace scaffold bodies).

### M8 — `evaluation/metrics.py` + harness wiring

- State metric(s) on `Z_hat` — trajectory diagnostics.
- Prediction metric(s) on `X_pred` — per ADR 0004: two residual streams (raw + reference), five stats each (RMSE, MAE, normRMSE, normMAE, Spearman), per-subject aggregation, three-way cohort reporting (train / test / val), τ-bins `1–3, 4–7, 8–10, 11+`, pooled-only top-speed subset gated by per-athlete Riegel fit.
- `run_evaluation` wires everything end-to-end; returns `EvalResult` keyed by `subject_id` with cohort labels.
- `evaluation/deconfounding.py` body (`project_to_reference`) is a hard dependency of the reference-space stream; lands before or alongside M8.

**Hyperparameters surfaced by name:** τ-horizons (bins are fixed per ADR 0004); metric target (`X`-channel, deferred to M9/M10 per late-naming).

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
| D6 | Functional form of `p(X | Z, P, E)` | M4 | ✂ dissolved — framework picks no form; each reference impl carries its own ADR (see Completed: sweep-harness signatures `cb81648`, `d7dc83e`) |
| D7 | Functional forms of `f` and `g` | M5 | ✂ dissolved — same as D6 |
| D8 | `StateEstimator` family | M6 | ✂ dissolved — same as D6 |
| D9 | Dimensionality of `Z` (`d_Z`) | M4 (load-bearing downstream) | ⏸ open — declared by each reference impl |
| D10 | Prediction metric(s) and τ-horizons | M8 | ✅ taken — ADR 0004 (`ba6f832`) |

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

- **Picking a privileged functional form.** The framework commits to no single form for observation, `f`, `g`, or estimator. Reference implementations are additive and each carries its own ADR; the sweep harness (`run_sweep`) composes them. D6/D7/D8 are dissolved, not deferred.
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
