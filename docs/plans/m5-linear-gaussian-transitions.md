# M5 — Linear-Gaussian transitions

**Status:** active
**Last updated:** 2026-04-24

Sequencing plan for M5 — first reference `WorkoutTransition` + `RestTransition` pair, covering the four workstreams identified after three rounds of expert audit on the proposed linear-Gaussian framing. Scoped within the broader sequencing at `post-scaffold-sequencing.md`.

---

## Current state

### Completed

None yet.

### Active

W1 — M4 spec amendment: `d_Z` demoted to hyperparameter.

### Audits landed

- **senior-scientist** (2026-04-24): linear-Gaussian is a defensible first-pass superset of Banister. Concerns flagged: `H^n` stationarity for long rests (cheapest fix: stratified rest-day residual diagnostic), hr_drift potentially being stimulus-side, asymmetric gain/loss not representable in linear `F`.
- **identifiability-auditor** (2026-04-24): partially identified. Unanimous asks: spectral-radius constraint on `H`; `r(n)` and `Q_rest(n)` derived from `H`; `H^n` form identifies `H` from n=1 data, n≥2 as mis-specification check. Flagged `π_stim` effective rank <5 as needing cohort-level rank reduction (deferred: kept as invalidation rather than pre-fit commitment).
- **master-statistician** (2026-04-24): derived-form `r(n)`/`Q_rest(n)`/`H` via discrete Lyapunov; `ρ(H) < 1` constraint; `d_Z` under-parameterization explicit; M6 estimator must pass posterior covariance (not plug-in mean) through EM.
- **identifiability-auditor plan-audit** (2026-04-24): plan faithful to the three audits after three fixes — M6 posterior-covariance commitment moved out of M5 coherence into deferred-to-M6 section, rest-bound-boundary residual bullet added, `π_stim` rank diagnostic elevated to M6 acceptance criterion.

---

## Constraints

- First reference impl of `f` and `g` Protocols in `statepace/transitions.py`.
- Single spec at `docs/reference_impls/linear-gaussian.md` covering both.
- `d_Z` is a sweep-time hyperparameter, not a fixed value; M4 spec and impl must be updated to match before M5 ships.
- Spec must honor all unanimous audit asks: `ρ(H) < 1` via `H = I − A` with `A` PSD; `r(n)` and `Q_rest(n)` derived from `H` and one-day parameters; `H^n` iterated form committed with n≥2 as a mis-specification check.
- M6-side asks (posterior covariance, rank diagnostic, A9-leakage diagnostic, within-cohort-half diagnostic) recorded here but not implemented at M5.

---

## Milestones

```
W1  M4 spec amendment: d_Z demoted to hyperparameter       ⏸ active
W2  M4 impl update: d_Z as constructor argument             ⏸ gated on W1 (commits together)
W3  M5 spec draft: linear-gaussian reference impl           ⏸ gated on W2
W4  M5 impl + tests: LinearGaussian class(es)               ⏸ gated on W3
```

### W1 detail — M4 spec amendment

Edit `docs/reference_impls/riegel-score-hrstep.md` §Family:
- `d_Z: 4` → `d_Z`: hyperparameter surfaced by name.
- Add invalidation bullet: sweep-optimal `d_Z` does not certify physiological interpretability of `Z` dimensions.

### W2 detail — M4 impl update

Edit `statepace/observation.py`:
- `RiegelScoreHRStep.d_Z` class attribute → `__init__(self, d_Z: int)` instance attribute.
- Confirm parameter shapes remain consistent with the passed `d_Z`.

Edit `tests/test_observation_riegel.py`:
- `RiegelScoreHRStep()` → `RiegelScoreHRStep(d_Z=4)` (preserve current test behavior).

Commit W1 + W2 together.

### W3 detail — M5 spec draft

Create `docs/reference_impls/linear-gaussian.md`. Covers both `f` and `g` in one spec. Required sections per `docs/reference_impls/README.md`. Key content:

- **Family:**
  - `f`: `Z_t = F · Z_{t-1} + G · π_stim + m + ν`, `ν ~ N(0, Q)`.
  - `g`: `Z_t = H^n · Z_{t-1} + r(n) + ν`, `ν ~ N(0, Q_rest(n))`.
  - Derived forms: `r(n) = (I − H^n)(I − H)⁻¹ r_1`; `Q_rest(n) = Σ_{k=0}^{n−1} H^k Q_1 (H^k)ᵀ`.
  - `H` parameterized as `H = I − A` with `A` PSD.
  - `d_Z`, `max_consecutive_rest_days` surfaced as hyperparameters.
- **Invalidation:** flat `G` directions across athletes; non-zero `cov(ν̂, ε̂)`; `Z`-residuals depend on lagged `hr_drift`; n≥2 rest-day residuals systematic; rest-bound-boundary residuals monotone; §8 loads beyond fitness/fatigue/trait/trend; `F`/`G` collinearity via selection.

### W4 detail — M5 impl + tests

Implementation in `statepace/transitions.py` conforming to `WorkoutTransition` + `RestTransition` Protocols. Tests in `tests/test_transitions_linear_gaussian.py`. Fixture: M4 fixture supplies `Channels`; synthetic `Z`-trajectory needed for `step`/`log_prob` tests — extend or add fixture at dispatch time.

---

## Open decisions

| # | Decision | Needed before | Status |
|---|---|---|---|
| W-D1 | One combined class vs two separate classes (`WorkoutTransition` + `RestTransition`) | W4 dispatch | ⏸ open — defer to dispatch |
| W-D2 | Fixture strategy for synthetic `Z`-trajectory (extend M4 fixture vs new factory) | W4 dispatch | ⏸ open — defer to dispatch |

No modeling-level open decisions; expert audits closed the family.

---

## Verification

| W | Verification |
|---|---|
| W1 | Spec edit reviewed; commit-readiness clean. |
| W2 | `pytest tests/ -v`: 24/24 passes with `RiegelScoreHRStep(d_Z=4)` explicit. |
| W3 | Spec passes `docs/reference_impls/README.md` section-required structure; commit-readiness clean. |
| W4 | `pytest tests/ -v` passes; new tests cover `step` shape, `log_prob` shape, `H^n` consistency at n=1, rest-bound guard, deterministic-given-seed. |

---

## Out of scope

- `π_stim` rank reduction as a pre-fit spec-side commitment. Kept as invalidation; addressed only if the symptom appears at M6.
- Asymmetric gain/loss and supercompensation: future reference impl.
- Independent-per-n `H(n)` parameterization: rejected on identifiability grounds.
- Shared `Q` across regimes (workout vs rest): independent `Q` and `Q_1` by default; cross-regime covariance is a diagnostic, not a parameter.
- `StateEstimator` (M6), `run_sweep` body (lands at M6), `forward.py`/`predict.py` glue (M7).

## Deferred to M6 (entry criteria for the estimator reference impl)

- Estimator consumes posterior `(mean, cov)` of `Z`, not plug-in posterior mean.
- `π_stim` rank diagnostic at M6 acceptance: flat `G` directions / sign-flipping `G` rows check. If triggered, `π_stim` needs cohort-level rank reduction.
- `cov(ν̂, ε̂)` diagnostic at M6 acceptance: non-zero signals A9 violation.
- Within-cohort weak-identification diagnostic (ADR 0002 follow-up): fit on random cohort halves, compare inferred `Z` on shared athletes.

---

## Maintenance

- At W1/W2/W3/W4 completion: move the workstream from Active → Completed; record the commit SHA in Current state.
- At any new audit: add a line under "Audits landed" with date and one-sentence finding.
- On close of M5: flip `Status` to `complete`; update `post-scaffold-sequencing.md` M5 milestone row with a pointer to this plan.
