# Linear-Gaussian retrospective

**Date:** 2026-04-24
**Scope:** retrospective on assumptions and limitations of the linear-gaussian reference impl; not a revision trigger.

Spec: `docs/reference_impls/linear-gaussian.md`
Plan: `docs/plans/m5-linear-gaussian-transitions.md`

## Assumptions this impl commits to

1. **Linearity of `f` in `(Z_{t-1}, π_stim)`.** Chosen as the minimal first-pass superset of Banister, closed-form under OLS and amenable to EM. Rules out asymmetric gain/loss (a single `F` cannot encode faster-decay-than-build), supercompensation, and any saturation in the stimulus response.
2. **Additive, state-independent Gaussian `ν`.** Chosen for closed-form residual covariance `Q`. Rules out heteroscedastic process noise (e.g., noise scaling with load or with current fitness).
3. **Stationary rest via `H^n`.** Chosen because it identifies `H` from n=1 data alone and keeps the parameter count flat across rest-run lengths. Rules out regime changes during long rest blocks (detraining, illness, taper-different-from-recovery).
4. **`H = I − A`, `A` symmetric, `0 < λ(A) ≤ 1`.** Chosen to enforce monotone contraction without oscillatory modes or unit roots. Rules out cross-dimensional coupling that requires non-symmetric `H` and rules out any `Z`-dimension with zero decay (pure trait components).
5. **Independent `Q` (workout) and `Q_1` (rest).** Chosen to keep the regimes parameter-disjoint. Rules out a shared driver of both regimes; cross-regime covariance lives only as a post-hoc A9 diagnostic.
6. **`π_stim` fixed to the 5-component set from `riegel-score-hrstep`.** Chosen for coherence with the paired observation impl. Rules out any stimulus channel (e.g., HRV, sleep) not declared on that impl.
7. **`d_Z` as a hyperparameter, not a physiological count.** Chosen so the sweep harness owns dimensionality selection end-to-end. Rules out a fixed "fitness + fatigue + …" decomposition and pre-commits that sweep-optimal `d_Z` carries no interpretability guarantee.
8. **OLS on plug-in `Z_prev.mean`.** Chosen because M5 Protocols take `Z` as input and the estimator-side posterior covariance lands at M6. Rules out propagating `Z_prev` uncertainty through the transition fit at M5.
9. **`H` identified from `n=1` only; `n ≥ 2` held out.** Chosen to respect the `H^n` functional form and use longer rests as a mis-specification check rather than as identifying data.

## Ranked next-iteration priorities

Ranked by likelihood-of-biting × cost-to-fix-in-next-iter.

### Top tier

1. **`π_stim` effective rank < 5 in cohort-scale fits.** Flagged by identifiability-auditor (concern 1). The five load components covary heavily across sessions (elevation gain and hr_load; step_load and hr_load). `G` rows flip sign across cohort halves with similar residuals. Remediation: cohort-level rank reduction on `π_stim` (PCA on training-window load vectors, pass rank-reduced stimulus into `G`); surface the retained-rank as a hyperparameter. Cheap — pre-fit projection, no Protocol change.
2. **Shared `Z_prev` plug-in between observation and transition fits.** Flagged by master-statistician (§5). The M5 impl uses `Z_prev.mean` in both `f`'s OLS and the rest OLS; the same point estimate enters `A·Z` in the observation impl. Measurement error in `Z_prev` is correlated across the two fits, biasing `F` toward identity. Remediation: the M6 estimator must pass full posterior `(mean, cov)` through EM; this is already in the M6 deferred list but should be the first thing the next transition impl can consume. Cost: Protocol-signature extension for `log_prob` only.
3. **`F`/`G` collinearity via selection on workout days.** Flagged by identifiability-auditor (concern 2). Athletes self-select into workouts such that `π_stim` and `Z_{t-1}` covary (fit athletes do harder sessions). Workout-day data identifies `F·Z + G·π_stim` jointly but not the split. Remediation: a parameter-swap likelihood test at M6 acceptance; if flat, require a deconfounding term (athlete-level baseline in `Z` pinned by rest-day data) or explicit prior on `G` shared across athletes. Cheap to diagnose, moderate to remediate.

### Mid tier

4. **`Q` rank-deficient at small `d_Z`.** Flagged by master-statistician (§2). With `d_Z=4` and a single residual stream, sample `Q` can hit rank deficiency under small-cohort fits; `multivariate_normal(..., allow_singular=False)` will raise. Remediation: diagonal-loading `Q` with a small ridge, or surface a `q_ridge` hyperparameter; alternatively switch to `allow_singular=True` with pseudo-inverse log-density. Cheap; known mitigation.
5. **`H^n` stationarity breaks over long rests.** Flagged by senior-scientist and master-statistician. Past ~7–10 consecutive rest days, the physiology transitions into detraining, which the stationary `H` cannot track. Remediation: the invalidation check (n≥2 residual sign test) is already scoped; the next impl can either shrink `max_consecutive_rest_days` tighter, or introduce a second rest regime past a threshold and splice. Moderate cost.
6. **`d_Z=4` vs §8 load accounting.** Flagged by master-statistician (§4). The framework §8 decomposition suggests ≥4 load families (fitness, fatigue, trait, trend); `d_Z=4` leaves no slack for cohort-level drift or measurement-structure capture. Remediation: at M6 the sweep should probe `d_Z ∈ {4, 6, 8}` and report residual reduction; family choice reopens if `d_Z` selection drifts to the upper end.

### Low tier

7. **`hr_drift` routed to `π_obs` rather than `π_stim`.** Flagged by senior-scientist. If within-session drift is driven by prior-load more than by current-state, it belongs on the stimulus side. Remediation: lagged-`hr_drift` residual diagnostic on `Z`-residuals at M6; if triggered, this is an ADR 0005 revisit, not a transition-impl change. High cost (DAG-level), lower probability of firing.
8. **Asymmetric gain/loss.** Flagged by senior-scientist. Linear `F` cannot represent faster-build-than-decay or the reverse; a sign-conditional residual diagnostic can detect it. Remediation: a non-linear or piecewise-linear `f` family — this is a new reference impl, not a next iteration of this one. High cost.
9. **Cross-athlete pooling amplifies ADR 0002 weak-identification.** Flagged by identifiability-auditor (concern 6). Remediation: cohort-half diagnostic already on M6 deferred list.
10. **Supercompensation.** Flagged by senior-scientist. Requires a non-monotone response to stimulus that the linear-Gaussian family cannot express at all. Family-change concern, not a next-iteration priority for a linear impl.

## Known compromises kept as invalidation conditions

These live in the spec's "What would invalidate this"; this retrospective elevates their visibility:

- Flat/sign-flipping `G` directions (priority 1 above).
- `Z`-residuals depending on lagged `hr_drift` (priority 7).
- Stratified n-rest-day residuals systematic for n ≥ 2 (priority 5).
- Rest-bound-boundary residuals monotone in n (priority 5, boundary form).
- `F`/`G` collinearity via selection (priority 3).

If any of these fire at M6, the next-iteration priorities above become live revision triggers.

## Out of scope for the retrospective

- Any change to the current `linear-gaussian` impl — the impl is committed and this document does not trigger revision.
- Estimator-side concerns (M6): posterior covariance propagation, EM convergence, gauge-fixing across `A, F, G, H` — tracked in the M5 plan's "Deferred to M6" section.
- New ADRs — this is documentation, not a decision record.
- Forward-simulation and prediction (M7): not a transition-impl concern.
- Upstream extraction quality for `π_stim` components (elevation, heat, load math) — observation-impl domain.
