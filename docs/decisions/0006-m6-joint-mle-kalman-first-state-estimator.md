# 0006: M6 first `StateEstimator` — joint MLE via differentiable Kalman, with structural priors

**Date:** 2026-04-24
**Status:** accepted

## Decision

The first reference `StateEstimator` is [`joint-mle-kalman`](../reference_impls/joint-mle-kalman.md): joint maximum-likelihood fit of the full state-space model by SGD on the cohort negative log-likelihood, with a differentiable Kalman filter integrating over `Z`. Paired with `riegel-score-hrstep` (M4) and `linear-gaussian` (M5); their `log_prob` is consumed inside the joint filter, their standalone `fit` is not invoked.

Structural priors are baked into the parameterization — the optimizer cannot escape them — and are research commitments that successor reference impls test by relaxation, one at a time:

- `d_Z = 4` with fixed `τ = (1, 7, 28, 84)` days, symmetric across `F` and `H`.
- `F` diagonal entries fixed by `τ`; off-diagonals bounded.
- `H` fully diagonal (passive decay per dimension); `r_1 = (I − H)·b` derived.
- Cohort-shared detraining target `b`.
- Bounded stimulus loading `G`.
- PSD-Cholesky covariances.
- Per-athlete `Z_0^{(i)}` as fit parameter (training and validation cohorts both).
- `π_stim` cohort-mean centering, frozen at fit-time.

`StateEstimator.infer` is widened to return a `ZPosterior` (sealed ABC); M6 returns `GaussianZPosterior` (mean + cov). Successor estimators define their own subclasses without re-widening the Protocol.

Initialization is staged (PCA → OLS → small-random) as a deterministic SGD start. Residual gauge is canonicalized post-hoc via a sign-convention cascade (`b_i ≥ 0` → `m_i ≥ 0` → declared norm of `G_{i·}` → boundary flag).

The sign convention applies to fitted parameters for diagnostics only; it imposes no constraint on the optimizer and commits no project-wide assignment of `Z`-axes to physiological quantities (CLAUDE.md late-naming rule).

This is a constrained-MLE first reference, not a generic linear-Gaussian baseline.

## Dimension naming

Dimensions are referred to as `Z_1, Z_2, Z_3, Z_4` indexed by `τ_i`. Coaching-language correspondence (day / week / month / quarter) is suggestive, not committed. Naming commitments are deferred to M9/M10.

## Context

ADR 0001 committed to shared parameters with between-athlete variation absorbed into per-athlete `Z`. ADR 0002 fixed the Protocol shape. M4 and M5 landed reference impls of the observation model and transitions. This ADR records the cross-cutting decisions for the first `StateEstimator` that ties them together. Audit history and the full parameterization live in `docs/plans/m6-state-estimator.md` and the spec.

## Alternatives considered

- **Joint EM (E-step Kalman smoother, M-step closed form).** Rejected: closed-form M-step does not scale cleanly to per-athlete `Z_0^{(i)}` as joint parameters, gives weak control over the optimization landscape (multi-start, LR schedules), and does not generalize to successor families (variational, non-Gaussian) that lack closed-form M-steps. SGD on the joint marginal likelihood scales on all three axes.
- **Mean-only `infer` return.** Rejected: the Kalman pass computes the covariance regardless; throwing it away and re-deriving in M7's `predict.py` is strictly worse than widening the Protocol once.

## DAG implications

- No edges move. ADR 0005's `π_obs`/`π_stim` split is satisfied structurally: M4 consumes `π_obs`, M5 consumes `π_stim`, the joint log-likelihood is their sum.
- A7 and A10 leaned on at full strength per ADR 0001/0002. A8 honored via large `Z_0^{(i)}` prior covariance.
- No project-wide channel-assignment commitments at M6.

## Empirical grounding

None. M6 validates on synthetic only (`make_linear_gaussian_cohort`). First empirical check against real-cohort dynamics is M1b / M10. Tier-1 acceptance is parameter recovery within empirical replicate-variance tolerance over `K = 20` synthetic replicates.

## Successor relaxations

Each structural prior has a named successor that tests it.

| # | Structural prior | Successor that relaxes it |
|---|---|---|
| 1 | Cohort-shared `(b, m)` | Hierarchical per-athlete `(b^{(i)}, m^{(i)})` (must relax both — they are dual rest-target / workout-drift) |
| 2 | Diagonal `H` | Free off-diagonal `H` |
| 3 | Fixed `τ = (1, 7, 28, 84)` | Free `τ_i` per dimension, or finer fixed schedule with extended train window |
| 4a | Linear-Gaussian observation likelihood | Heavy-tailed / non-Gaussian observation likelihood |
| 4b | Gaussian posterior on `Z` | Variational estimator with non-Gaussian posterior |
| 5 | Stationary `(F, G, H, b)` (A10) | Switching SSM with regime indicators |
| 6 | 210-day train window | Extended train window (M1b real-data scope) |

Tier-C invalidation conditions tie 1:1 to rows 1, 2, 3, 5, 6.

## What would invalidate this

- **Tier A — re-open impl (W4):** `d_Z` mismatch; rest-bound overrun leaking into the loss; PSD drift; non-determinism under fixed seeds; multi-start divergence under sign convention as pathology; Tier-1 W6 failure.
- **Tier B — re-open this ADR (W3):** staged init insufficient; differentiable-framework dependency blocked; learning rate / multi-start count miscalibrated against the K = 20 study.
- **Tier C — close M6 with finding; open successor:**
  - Linear-Gaussian family inadequate on M10.
  - Regime-shift contamination of shared `θ` (W6 Tier-2 per-athlete vs shared-`θ` pseudo-likelihood ratio).
  - Quarter-scale ceiling: `τ_4 = 84` insufficient for chronic adaptation on M10 (train window doesn't identify longer modes).
  - Cohort-shared `b` empirically binding: between-athlete equilibrium-variance exceeds operational threshold; fires successor 1 before M10.

## Follow-ups

- W4 lands the impl, `ZPosterior` ABC + `GaussianZPosterior`, and scaffold signature updates in `forward.py` / `predict.py` (bodies are M7).
- W5 wires `run_sweep` and `run_evaluation`; `run_evaluation` consumes `ZPosterior` for calibrated prediction intervals.
- W6 produces Tier-1 + Tier-2 diagnostics + parameter-count sanity report; senior-scientist interprets before M6 closes.
- On M6 close: record successor milestone (hierarchical `Z_0^{(i)}` + per-athlete `(b^{(i)}, m^{(i)})`) in `post-scaffold-sequencing.md`.
