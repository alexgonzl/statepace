# M6 — First reference `StateEstimator`

**Status:** active (W5 closed; W6 unblocked)
**Last updated:** 2026-04-24

Sequencing plan for M6 — first reference `StateEstimator` paired with the `riegel-score-hrstep` observation (M4) and `linear-gaussian` transitions (M5). Also lands the two harness bodies deferred from M4: `run_sweep` and `run_evaluation`.

M6's estimator is a **constrained maximum-likelihood fit of the joint state-space model** by gradient descent on the negative log-likelihood, with a differentiable Kalman filter integrating over the latent `Z`. The structural priors (multi-timescale decomposition, diagonal rest, cohort-shared detraining target, headroom-bounded stimulus) are baked into the parameterization; the optimizer cannot escape them. This is **not** the most permissive linear-Gaussian fit — it is the most physiologically-disciplined one consistent with the framework's `Z`-structure. Successor reference impls relax structural priors one at a time to triangulate which priors are load-bearing. Initialization follows the standard system-identification recipe: PCA of warmup X̃ for `A`, OLS of X on `(P, E)` for `B, C, d`, near-identity for `F`, small for `Q, Σ`, diffuse for `Z_0`.

Structural constraints baked into the parameterization:

- **Multi-timescale `Z`:** `d_Z = 4` with fixed per-dimension time constants `τ = (1, 7, 28, 84)` days applied symmetrically to workout (`F`) and rest (`H`) regimes. Dimensions are referred to as `Z_1..Z_4` indexed by `τ_i`. Each `τ_i` plausibly maps to a coaching-recognized scale (day / week / month / quarter), but this correspondence is *suggestive*, not committed — see W3 ADR for the late-naming policy.
- **`F` structure:** diagonal entries `F_ii = exp(−1/τ_i)` fixed by schedule; bounded off-diagonals `F_ij = α_ij · (1 − F_ii)`, `α_ij = tanh(β_ij)` free.
- **`H` structure:** fully diagonal, `H_ii = exp(−1/τ_i)`. Rest is passive decay per dimension — no cross-dimension coupling (first principles).
- **Cohort-shared detraining target `b`:** rest dynamics decay toward `b` per dimension, not toward zero. `r_1 = (I − H)·b` derived.
- **Bounded stimulus loading `G`:** `G_ij = (1 − F_ii) · g_ij`, `g_ij` free real-valued. Sign unrestricted (stimulus can push `Z` in either direction).
- **PSD covariance parameterizations** `Σ = L_Σ L_Σ^T` etc.
- **`B, C, d, m, Z_0^{(i)}` unconstrained.**

`StateEstimator.infer(...)` returns **`(mean, cov)` posterior trajectory** (widened from scaffold's mean-only). This is a small Protocol widening landing at M6; it lets M7's `predict.py` consume calibrated uncertainty without re-running the filter.

This estimator is a **constrained-MLE first reference**, not a generic-baseline benchmark. The structural priors are research commitments that successor families test by relaxation.

---

## Current state

### Completed

- **W1** — M6 spec draft (`docs/reference_impls/joint-mle-kalman.md`). Commit `a6e7661`.
- **W2** — Spec audit rounds. Audit-clean across identifiability-auditor and senior-scientist (round 8). Commit `a6e7661`.
- **W3** — M6 ADR (`docs/decisions/0006-m6-joint-mle-kalman-first-state-estimator.md`). Commit `09552b6`.
- **W4** — M6 impl + tests. `JointMLEKalman` in `statepace/filter.py`; `ZPosterior` sealed ABC + `GaussianZPosterior`; signatures widened in `forward.py` and `predict.py`; `architecture_map.md` updated; 25 new tests; 57 passing total in 88s. Three rounds of architect-driven remediation (env, rest-bound plumbing, NA1 recovery test + fixture rebalance, scope-creep removal). Spec note `W-D-internal-val` closed (commit `a460bc9`). Commit `10e8d12`.
- **W5** — `run_evaluation` and `run_sweep` bodies + tests. Schema widening (commit `69e1875`): `EvalResult.Z_hat` to `ZPosterior`, new `XPredictive(mean, samples, n_samples)` for sample-based observation-space predictives. Body implementation (commit `e7d3c22`): 8 new harness tests, 65 passing total in 5:23. Architect-driven contract fix: added `StateEstimator.fitted_observation() -> ObservationModel` to the Protocol surface, eliminating the need for the harness to reach into estimator-private state to populate the observation instance for predictives.

### Active

W6 — Acceptance diagnostics. Tier-1 (parameter recovery on K=20 replicate study; posterior coverage calibration; multi-start consistency; validation `μ_0` recovery). Tier-2 family-adequacy diagnostics (`π_stim` effective rank, `cov(ν̂, ε̂)`, within-cohort weak identification, per-athlete vs shared-`θ` pseudo-likelihood ratio for regime-shift, between-athlete equilibrium-variance for cohort-shared-`b` triggering). Parameter-count sanity report. M6 closes on clean Tier-1; Tier-2 numbers populate the close-out findings.

### Audits landed

- **identifiability-auditor rounds 1, 3, 4** (2026-04-24): drove the pivot away from joint-EM, exposed bootstrap circularity, then dissolved mechanical asks under SGD on joint marginal likelihood.
- **senior-scientist rounds 2, 4** (2026-04-24): drove pivot to Kalman baseline, then flagged `Z ≠ framework-Z` under PCA gauge. Resolved under SGD.
- **9-option menu audits** (2026-04-24): both auditors converged on "joint factor model on warmup" (option 8); user pivoted further to SGD-on-joint-likelihood per literature survey.
- **Literature survey** (2026-04-24): staged init (PCA → OLS → small-random) + SGD on joint marginal likelihood is the textbook treatment.
- **identifiability-auditor round 5** (2026-04-24): four narrow asks — NA1 (validation `Z_0` recovery test), NA2 (gauge canonicalization), NA3 (M6/M5 `H` parameterization coherence), NA4 (rest-bound boundary rule). NA2 and NA3 subsequently dissolved by fixing `τ` schedule and making `H` diagonal.
- **senior-scientist round 5** (2026-04-24): no blockers. Narrow asks — S1 (regime-shift as Tier-C), S2 (name concrete successor), S3 (decide `infer` covariance explicitly), S5/S8 (rename Tier-1, add late-naming sentence), S6 (Tier-1 tolerances from CRB), S7 (EM-vs-SGD justification in ADR).
- **identifiability-auditor round 6** (2026-04-24): all four NAs closed by structural-constraint pivot. Three new mechanical asks — U1 (center `π_stim` to dissolve `m`/`G·E[π_stim]` trade-off), U2 (`G_ij = γ_j · (1 − F_ii) · σ_ij` parameterization has `γ`/`σ` scaling redundancy; drop `γ_j`, parameterize `G_ij = (1 − F_ii) · g_ij` directly), U3 (sign-convention `b_i ≥ 0` doesn't canonicalize when `b_i ≈ 0`; cascade fallback to `m_i`, then `G`-norm). Two non-unanimous: C1 (slow-row off-diagonals `α_4j` near noise floor; declare wide tolerances), C3 (per-athlete rest-density to parameter-count sanity report).
- **senior-scientist round 6** (2026-04-24): no blockers. Two strong: S-S1 (drop "real ML estimator, not deliberately-biased baseline" framing — structural priors changed it; reframe as constrained-MLE first reference), S-S2 (cohort-shared `b` is dubious for mixed-ability cohorts; if relaxed, must also relax `m` to per-athlete `m^{(i)}` — they're dual). Four moderate: S-M3 (de-name dimensions in W3 ADR; refer as `Z_1..Z_4` until M10 confirms named-construct mapping), S-M4 (`ZPosterior` should be sealed ABC with `GaussianZPosterior` for M6, future subclasses for non-Gaussian successors), S-M5 (W3 ADR successor "table" with one row per structural prior), S-M6 (rename "CRB-derived" tolerances to "empirical replicate-variance"; specify K and per-parameter normalization).
- **PM resolution** (round 6, 2026-04-24): all eight asks (U1, U2, U3, S-S1, S-S2, S-M3, S-M4, S-M5, S-M6) folded in. None re-open family or structural constraints; all are local fixes to parameterization, framing, or W6 diagnostics.
- **identifiability-auditor round 7** (2026-04-24): four spec asks (V1 `π̄_stim` freeze; V2 flag-propagation policy; V3 multi-start metric exclusion; V4 rest-density exclusion). All folded in.
- **senior-scientist round 7** (2026-04-24): four concerns (C1 dimension de-naming seams; C2 verification table CRB-derived rename; C3 3× threshold derivation/operational framing; C6 successor row 4 conflation). All folded in.
- **identifiability-auditor round 8** (2026-04-24): **audit-clean.** V1–V4 verified folded in correctly.
- **senior-scientist round 8** (2026-04-24): **audit-clean.** C1, C2, C3, C6 verified folded in correctly.
- **sports-science-advisor** (2026-04-24): recommended `τ = (1, 7, 42, 180)` on physiological grounds; flagged `τ_rest = τ_workout` as right at first impl; 5-stimulus-vector reasonable with time-in-zone as most important missing component.
- **identifiability-auditor on `τ` schedule** (2026-04-24): `τ_max = 180` identifiability-toxic with 210-day train window (only 1.17 e-foldings). Recommended `(1, 4, 16, 64)` factor-4. Flagged uniform log-spacing gives uniform pairwise correlation (evenly distributes off-diagonal identifiability); "Fourier basis" analogy overstated but structural property (self-similar scale-equivariant basis) is real.
- **PM resolution on `τ` schedule (2026-04-24):** `(1, 7, 28, 84)` factor-3.5 — physiologically named (day/week/month/quarter) and identifiability acceptable (τ_4 = 84 → 2.5 e-foldings, loose but not toxic).
- **PM resolution on `infer` covariance (S3):** widen Protocol to return `(mean, cov)` at M6. Covariance is computed by the Kalman filter regardless; throwing it away and re-deriving at M7 is strictly worse.
- **senior-scientist literature-fit assessment** (2026-04-24, post-W3): position M6 as a *constrained NLME fit of a multi-component fitness-fatigue model*, not as a novel SSM. Banister/Busso/Hellard is the fitness-fatigue lineage; Pinheiro-Bates / Sheiner-Beal NLME is the cohort-shared-`θ` + per-athlete-`Z_0` lineage. M6 sits at the most restrictive end of the NLME spectrum (only `Z_0` random); successor row 1 is the natural NLME next step. **Action deferred to M6-close:** add a one-paragraph literature anchor to ADR 0006 *Context* citing Hellard 2006 (Banister + hierarchical Bayes) and Pinheiro & Bates 2000 (NLME). Differentiable Kalman + SGD treated as plumbing (no positioning needed). Structural-prior-relaxation strategy stands as research design; no formal label fits cleanly.

Audit rounds continue as W1 iterates on this draft.

---

## Constraints

- First reference impl of the `StateEstimator` Protocol in `statepace/filter.py`.
- Single spec at `docs/reference_impls/<estimator-slug>.md`. Paired at bundle-construction time with `riegel-score-hrstep` (M4) and `linear-gaussian` (M5).
- **Maximum-likelihood fit by gradient descent on the joint negative log-likelihood.** Differentiable Kalman filter computes `log p(X | θ, Z_0^{(i)})` per athlete; cohort-level loss is sum over athletes. Optimizer (Adam or similar) updates all of `θ = (β, τ's only via log — or fixed; b, m, G, B, C, d, Σ, Q, Q_1, ...)` and per-athlete `Z_0^{(i)}` simultaneously.
- **Structural constraints on parameterization:**
  - **`d_Z = 4`** at M6 default (sweep may vary if future ADR relaxes).
  - **`τ_workout = τ_rest = (1, 7, 28, 84)` fixed** in days. Dimensions referred to as `Z_1..Z_4`. Plausible mapping to day / week / month / quarter is suggestive only (W3 ADR late-naming policy). Not free parameters.
  - **`F`:** diagonal entries `F_ii = exp(−1/τ_i)` fixed; off-diagonals `F_ij = α_ij · (1 − F_ii)` for `i ≠ j`, with `α_ij = tanh(β_ij)`, `β_ij` free.
  - **`H`:** fully diagonal, `H_ii = exp(−1/τ_i)`. No free parameters — automatically satisfies M5's `0 < λ(H) < 1`.
  - **`b ∈ ℝ^4`:** cohort-shared, free (detraining target per dimension).
  - **`r_1 = (I − H)·b`:** derived from `b, H`, not separately fit.
  - **`m ∈ ℝ^4`:** free workout-day drift.
  - **`G_ij = (1 − F_ii) · g_ij`**, `g_ij` free real-valued. Sign unrestricted — stimulus can push `Z` in either direction per dimension. (U2: original parameterization with `γ_j · σ_ij` had a scaling redundancy; dropped to a single free `g_ij` per `(i, j)`.)
  - **`Σ, Q, Q_1`:** Cholesky-parameterized, `Σ = L_Σ L_Σ^T`, etc.
  - **`B, C, d, Z_0^{(i)}`:** unconstrained (standard linear regression on `(P, E)` for `B, C, d`; per-athlete initial state for `Z_0^{(i)}`).
- **`π_stim` centering** (U1). Before fit, compute the cohort-mean `π̄_stim` over training-cohort workout days; subtract from `π_stim` throughout. The cohort-mean offset is rolled into `m`'s init. This dissolves the practical `m` ↔ `G · E[π_stim]` trade-off without adding a parameter or a prior. **`π̄_stim` is computed once on the training cohort, frozen at fit-time, and re-applied (not re-computed) at validation-cohort `infer` and at every subsequent `infer` call** (V1). Re-computing on a different cohort would shift `m`'s intercept basis and bias validation `Z_0^{(i)}` recovery.
- **Staged initialization** (init only; no constraint on optimizer thereafter):
  - `A`: PCA of warmup X̃ pooled across training cohort, top-`d_Z = 4` principal directions.
  - `B, C, d`: OLS regression of X on `(P, E)` over warmup.
  - `Σ`: residual covariance from the `(P, E)` regression.
  - `β_ij = 0` (initial `α_ij = 0`, so `F` starts diagonal).
  - `b`: small random near-zero.
  - `m`: cohort-mean offset from `π_stim` centering (U1) absorbed; otherwise small random near-zero.
  - `g_ij`: small random.
  - `Q, Q_1`: small diagonal (e.g., `0.01·I`).
  - `Z_0^{(i)}`: zero, with large prior covariance as fit parameter or fixed large per spec.
- **Multi-start.** Spec declares seed count (recommended 3–5); selection rule is highest validation log-likelihood. The fixed `τ` schedule + diagonal `H` dramatically reduce the gauge orbit, so multi-start consistency should be tight.
- **Per-athlete `Z_0^{(i)}` is a fit parameter.** Adds `N · d_Z` parameters. Validation-cohort `Z_0^{(i)}` is fit at `infer` time under frozen `θ`, by gradient descent on the single-athlete likelihood. Same code path as training `infer`.
- **`infer` Protocol widened to `ZPosterior`** (S-M4). `StateEstimator.infer(...) -> ZPosterior` where `ZPosterior` is a **sealed ABC** with concrete subclasses per posterior family. M6 returns `GaussianZPosterior` containing mean `(T, d_Z)` and cov `(T, d_Z, d_Z)`. Future successor estimators (variational, particle, switching SSM) define their own concrete subclasses (`SampleZPosterior`, `MixtureZPosterior`, etc.) without re-widening the Protocol. `forward.py` and `predict.py` dispatch on the concrete subclass type. The ABC defines a minimal interface (e.g., `mean()`, `sample(n)`, `marginal_log_pdf(z)`) that all subclasses implement; M7+ scaffold uses the ABC interface. This avoids locking the Protocol into Gaussian-only summaries while delivering the M6 widening cleanly.
- **Gauge residual.** The fixed `τ` schedule pins the diagonal-ordering gauge; `H` diagonal eliminates further rotation. Residual gauge: sign flips on individual dimensions (flipping `Z_i → -Z_i`, `A` row, `b_i`, `m_i`, `G_i·`) are still likelihood-invariant. **Post-hoc sign convention is a cascade** (U3): for each dimension `i`, declare positive sign by `b_i ≥ 0`; if `|b_i| < tol_b`, fall back to `m_i ≥ 0`; if both `|b_i|` and `|m_i|` are below tolerance, fall back to a declared norm of `G_{i·}` (e.g., signed sum); finally, leave a flag in the report identifying dimensions at the gauge boundary (none of the three primary tie-breakers fired above tolerance). Applied post-fit for diagnostics; not an optimizer constraint. Tolerances `tol_b`, `tol_m` declared in spec. **Flag-propagation policy (V2):** boundary-dimension flags are reported in the M6 fit artifact and consumed by M7+; what M7's `predict.py` should *do* with a flagged dimension (propagate, drop, or just record) is deferred to M7 scoping.
- **`Prior.diffuse=True` (§A8) honored** via large `Z_0^{(i)}` prior covariance. Warm-up masking at harness (boundary rule 6) unchanged; scoring still masks warmup.
- **Rest-day handling:** no observation update; filter propagates `Z_{t−1} → Z_t` under `H` with accumulated process noise. `(γ, s, b)` get gradient signal from cross-rest-gap `Z` consistency.
- **Rest-bound overrun (§A5):** spec declares the boundary rule — drop overrun rows from the loss; resume filtering at first post-gap workout-day observation under a declared re-entry prior. Harness flags via `EvalResult.rest_bound_violations`.
- **`d_Z` matching:** estimator's `d_Z` must equal paired impls' `d_Z`; construction-time error on mismatch.
- `run_sweep` and `run_evaluation` bodies land at M6.
- **ADR 0005 edge assignment** satisfied structurally: M4's likelihood consumes `π_obs`; M5's consumes `π_stim`; the joint log-likelihood is their sum.
- No project-wide channel-assignment commitments in M6 artifacts.
- Public API (`statepace/__init__.py`) unchanged; reference impls remain additive and non-exported.

---

## Workstreams

```
W1  M6 spec draft: SGD + structural constraints          ✅ a6e7661
W2  M6 spec audit rounds + redrafts                       ✅ a6e7661
W3  M6 ADR: family + init + gauge + constraints           ✅ 09552b6
W4  M6 impl + tests: differentiable Kalman + SGD          ✅ 10e8d12
    — includes Protocol widening (infer returns ZPosterior)
    — includes scaffold updates to forward.py / predict.py signatures
W5  run_sweep + run_evaluation bodies + tests             ✅ e7d3c22
    — includes EvalResult schema widening + XPredictive (69e1875)
    — includes StateEstimator.fitted_observation Protocol method
W6  Acceptance diagnostics                                 ⏳ active
```

---

## Workstream detail

### W1 — M6 spec draft

Create `docs/reference_impls/<slug>.md`. Slug TBD.

**Required spec content (per README):** standard 8 sections, with `## Family` carrying the full structural parameterization and `## Coherence with other impls` declaring that M4 and M5's `fit` methods are not invoked; their likelihoods are consumed directly via the joint Kalman filter.

**Load-bearing spec decisions:**

#### §Estimator family
- Loss form (per-day NLL summed over days + athletes); smoothing at fit time (RTS backward pass) vs filter-only.
- Optimizer (Adam recommended), learning rate schedule.
- Convergence criterion (max iterations + val-loss patience).
- Multi-start seed count.
- Differentiable-Kalman framework (PyTorch vs JAX — see W-D2).

#### §Structural parameterization
- Confirm `d_Z = 4` and `τ = (1, 7, 28, 84)` fixed; declare rationale.
- `F` parameterization as above.
- `H` parameterization as above.
- `G` parameterization as above.
- `b` as cohort-shared free.
- PSD Cholesky for `Σ, Q, Q_1`.
- Per-athlete `Z_0^{(i)}` as fit parameter.
- Post-hoc sign convention for residual gauge (`b_i ≥ 0`).

#### §Initialization
- Staged init per Constraints above.
- Random-seed handling: each multi-start seed re-randomizes the small-random components; PCA + OLS inits are deterministic given cohort.

#### §Numerical hygiene
- PSD parameterizations for covariances.
- Backprop-through-Kalman stability (Joseph form if needed).
- Numerical precision (float64 for Kalman pass recommended).

#### §Rest-bound and edge cases
- Rest-bound overrun: mask from loss; re-entry prior declared.
- Validation-cohort `infer` procedure (on-the-fly `Z_0^{(i)}` fit under frozen `θ`).

#### §`infer` Protocol and `ZPosterior` dataclass
- Define `ZPosterior` (mean `(T, d_Z)` + cov `(T, d_Z, d_Z)`).
- Define the return contract explicitly.
- Scaffold changes that must land: `statepace/filter.py` Protocol; `statepace/forward.py` and `statepace/predict.py` signatures consume `ZPosterior` instead of raw `Z`.

**Hyperparameters surfaced by name:** `d_Z`, `max_consecutive_rest_days`, learning rate, max iterations, patience, multi-start seed count, PSD-Cholesky regularization (if any), `Z_0` prior covariance, Tier-1 tolerance thresholds.

**Invalidation conditions — three tiers:**

- **Tier A (re-open W1):** `d_Z` mismatch; rest-bound overrun leaking into loss; PSD drift; non-determinism given fixed seeds; multi-start parameters diverge under sign convention (pathology, not local minima).
- **Tier B (re-open W3):** Init recipe insufficient (reliable non-convergence); PyTorch/JAX dependency adoption blocked; learning rate / multi-start-count clearly miscalibrated.
- **Tier C (close M6 with finding; open successor milestone):**
  - Local minima dominate (multi-start spread large under sign convention).
  - Validation-cohort test gap on M10 attributable to variance throw-away through mean-only infer (now moot — covariance surfaced — but retained as sanity condition).
  - Linear-Gaussian family inadequate for real-cohort dynamics on M10.
  - **Regime-shift contamination of shared `θ` (S1):** athletes with mid-window injuries, layoffs, or periodization re-points pull shared `θ` toward an average that fits neither pre-shift nor post-shift dynamics. Tier-2 diagnostic at W6 reports per-athlete `θ_i` fit on a small holdout vs shared-`θ` pseudo-likelihood ratio; large ratios flag A10 damage.
  - **`τ_4 = 84` dimension insufficient for chronic adaptation on M10.** Recorded: M6 baseline commits to quarter-scale (84-day) longest mode because train window (210 days) does not identify longer modes. If real-cohort dynamics require year-scale `Z` dimension, train-window extension or `d_Z` expansion is the successor path.

**Logical closure:** M6 closes on clean Tier-1 + clean Tier-A. Tier-B re-opens W3. Tier-C fires retrospectively.

**Dispatch:** Tier-2 specialist — joint draft by master-statistician (SGD/Kalman mechanics) and senior-scientist (structural-constraint rationale, successor-family framing).

### W2 — M6 spec audit rounds + redrafts

Unbounded rounds. **Exit only on audit-clean** across identifiability-auditor, senior-scientist, master-statistician, with cross-auditor re-audit on material redrafts.

### W3 — M6 ADR

Records the cross-cutting decisions:

- **Posture: constrained-MLE first reference (S-S1).** Not a generic linear-Gaussian baseline, not a deliberately-biased plug-in. A structurally-priored Kalman MLE. The structural priors are research commitments. Successor families test by relaxation, one prior at a time.
- **Dimension naming policy (S-M3).** ADR refers to dimensions as `Z_1, Z_2, Z_3, Z_4` with time constants `τ_1 = 1, τ_2 = 7, τ_3 = 28, τ_4 = 84` days. Physiological naming (acute-fatigue / weekly mesocycle / monthly training block / quarterly training phase) is a **suggestive correspondence** noted in commentary but not committed: it is what each `τ_i` *plausibly maps to* in coaching language, not what M6 *claims to estimate*. Naming commitments are deferred to M9/M10 per CLAUDE.md late-naming rule.
- SGD-on-joint-likelihood as the first reference family.
- **EM-vs-SGD justification (S7):** closed-form EM exists in NumPy; we choose SGD for (1) natural scaling to per-athlete `Z_0^{(i)}` as joint parameters, (2) direct optimization-landscape control (multi-start, LR schedule), (3) clean generalization to successor families (variational, non-Gaussian) that lack closed-form EM.
- Staged-init recipe.
- `π_stim` cohort-mean centering (U1).
- Structural constraints (`d_Z = 4`; fixed `τ = (1, 7, 28, 84)`; `F, H, G` parameterizations per Constraints; cohort-shared `b`).
- `infer` Protocol widening to `ZPosterior` (sealed ABC; M6 returns `GaussianZPosterior`).
- Post-hoc sign-convention cascade (U3) for residual gauge; late-naming sentence (S5): "The sign convention is applied to fitted parameters for diagnostics only; it imposes no constraint on the optimizer and does not commit any project-wide assignment of `Z`-axes to physiological quantities (CLAUDE.md late-naming rule)."
- Invalidation tier split, including regime-shift Tier-C (S1) and quarter-scale-ceiling Tier-C.

**Successor table (S-M5).** Each structural prior in M6 has a named relaxation that tests it. The M6 ADR enumerates them; it does not scope their plans.

| # | Structural prior | Successor that relaxes it | Diagnostic / question |
|---|---|---|---|
| 1 | Cohort-shared `(b, m)` | Hierarchical per-athlete `(b^{(i)}, m^{(i)})` (S-S2: must relax both — they are dual rest-target / workout-drift) | Does partial pooling on equilibrium pair beat cohort-shared on validation? |
| 2 | Diagonal `H` | Free off-diagonal `H` | Does cross-dimension rest coupling buy meaningful predictive performance? |
| 3 | Fixed `τ = (1, 7, 28, 84)` | Free `τ_i` per dimension (or a finer fixed schedule with extended train window) | Are the time constants identifiable from data? Are they near `(1, 7, 28, 84)`? **Sub-question:** does any dimension's fitted time constant diverge to infinity — i.e., one of the dimensions is effectively immutable on the training horizon? Per A7 (stable individual traits live inside `Z`), an identity-diagonal dimension with zero stimulus-loading row is the architectural form of a stable-trait dimension; whether to adopt it should be data-evidenced via this relaxation, not imposed upfront. |
| 4a | Linear-Gaussian observation likelihood (Gaussian X residuals) | Heavy-tailed / non-Gaussian observation likelihood (e.g., Student-t residuals) | Are Riegel-score / HR / step-load residuals heavy-tailed enough that Gaussian likelihood mis-weights outliers? |
| 4b | Gaussian process noise / Gaussian posterior on `Z` | Variational estimator with non-Gaussian posterior (`SampleZPosterior` or similar) | Does non-Gaussian posterior calibration improve M10 prediction intervals beyond what `GaussianZPosterior` provides? |
| 5 | Stationary `(F, G, H, b)` (A10) | Switching SSM with regime indicators | Are A10 violations (injury, layoff, periodization shift) the binding constraint on real cohorts? |
| 6 | 210-day train window | Extended train window (M1b real-data scope) | Do longer modes (`τ > 84`) become identifiable and improve prediction? |

The M6 ADR records this table; successor milestones are scoped separately. The Tier-C invalidation conditions tie 1:1 to rows 1, 2, 3, 5, and 6 of this table.

### W4 — M6 impl + tests

Implementation in `statepace/filter.py` (joint SGD + Kalman). Also scaffold edits:
- `statepace/filter.py`: widen `StateEstimator.infer` return type; add `ZPosterior` dataclass.
- `statepace/forward.py`: update `forward_state` signature to consume `ZPosterior`.
- `statepace/predict.py`: update `predict_session` signature accordingly.
- `statepace/channels.py`: no change to `Z` type alias (still `Array`); add `ZPosterior` alongside.
- `docs/architecture_map.md`: reflect widening.
- `statepace/__init__.py`: add `ZPosterior` to exports if appropriate.

**Test coverage:**
- `fit`: staged init deterministic; SGD loop deterministic under fixed seeds.
- `infer(channels)`: returns `ZPosterior` with correct shapes for training-cohort (frozen `Z_0^{(i)}`) and validation-cohort (on-the-fly `Z_0^{(i)}` fit).
- `Prior.diffuse=True` honored via large `Z_0` prior covariance.
- `d_Z` mismatch raises at construction.
- Rest-bound overrun masked.
- PSD covariances preserved across SGD.
- Multi-start deterministic under seed list; selection reproducible.
- Post-hoc sign convention idempotent.
- Validation `Z_0^{(i)}` recovery test on synthetic (NA1): generate cohort with known `Z_0^{(i)}`; fit on training subset; `infer` on held-out validation subset; recovered validation `Z_0^{(i)}` matches generator within declared tolerance.

**Dispatch:** focused-engineer.

### W5 — `run_sweep` + `run_evaluation` bodies + tests

Mechanical wiring. `run_evaluation` consumes `ZPosterior` from `infer`; passes posterior mean + covariance to observation.forward for calibrated prediction intervals at score-window days. `rest_bound_violations` populated by scanning runs.

**Dispatch:** focused-engineer.

### W6 — Acceptance diagnostics

#### Tier 1 — Optimizer correctness on in-family synthetic (renamed per S5/S8)

- **Parameter recovery on `make_linear_gaussian_cohort` synthetic.** Generate data from a linear-Gaussian SSM matching M6's structural constraints. Fit M6; apply post-hoc sign-convention cascade to recovered and generator parameters; compare. **Empirical replicate-variance tolerance (S-M6, replaces S6's "CRB-derived" framing):** generate K = 20 replicates of the synthetic cohort (declared in W-D11), run multi-start SGD on each, report empirical sampling distribution of recovered parameters. Set per-parameter Tier-1 tolerance at (mean MLE bias + 2 × empirical SD), with per-parameter normalization declared (recovered/generator ratio for positive scalars; absolute deviation in standardized units for unbounded; matrix Frobenius norm for blocks like `F`'s off-diagonals). Pass condition: "recovery within empirical MLE sampling noise." Slow-row off-diagonals (`α_4j` for `j ≠ 4`) are flagged as known-weak (C1): tolerance set wide and reported with a flag, not blocking pass.
- **Posterior coverage calibration, post-warmup.** 68/95/99% credible intervals on `Z_t` from the Kalman pass contain true `Z_t` at the corresponding empirical frequency. Pass: empirical coverage within declared tolerance of nominal.
- **Multi-start consistency under sign convention.** Across 3-5 seeds, sign-convention-applied parameters cluster within tolerance. **Known-weak parameters (per C1: slow-row off-diagonals `α_4j` for `j ≠ 4`) are excluded from the clustering metric** (V3); their natural noise-floor spread would otherwise confound the test. Boundary-flagged dimensions per U3 are also excluded. Spread outside tolerance on the remaining (non-excluded) parameters re-opens W4 (Tier A) or signals Tier C.
- **Validation-cohort `Z_0` recovery (NA1).** Generate synthetic cohort with known per-athlete `Z_0^{(i)}`; train on training-cohort subset; `infer` on validation-cohort subset; compare recovered validation `Z_0^{(i)}` to generator. Pass: within declared tolerance. **Athletes with rest-density < 5% are excluded from `b`-related recovery checks** (V4); for synthetic recovery tests where rest-density is generator-controlled, the generator setting is declared in spec.

#### Tier 2 — Family-adequacy diagnostics (recorded; no re-open at M6)

- `π_stim` effective-rank diagnostic.
- `cov(ν̂, ε̂)` diagnostic.
- Within-cohort weak-identification diagnostic (ADR 0002).
- **Per-athlete vs shared-`θ` pseudo-likelihood ratio (S1):** fit per-athlete `θ_i` on a small holdout; compare against shared-`θ` LL. Large ratios indicate A10 violations / regime-shift contamination.
- **Between-athlete equilibrium variance (S-S2 trigger):** for each athlete, extract the implied per-athlete equilibrium `Z_∞^{(i)} = (I − H)⁻¹ b` from late-rest states (using the smoothed posterior). Compare empirical between-athlete variance of `Z_∞^{(i)}` against the variance induced solely by per-athlete `Z_0^{(i)}` differences under shared `b`. The numeric trigger threshold is **operational, not derived**: if the between-athlete variance ratio exceeds an order-of-magnitude floor (provisionally 3×, declared as W-D15), this is taken as qualitative evidence that the cohort-shared `b` constraint is the binding limitation and successor 1 (per-athlete `(b^{(i)}, m^{(i)})`) fires before M10. The qualitative direction is load-bearing; the exact threshold is a heuristic and may be revisited via likelihood-ratio test in a successor iteration. Athletes with rest-density < 5% are excluded from this diagnostic per V4 (the implied `Z_∞^{(i)}` is not informative without late-rest states).
- (Optional) **Fisher information on recovery parameters (S4):** report inverse-Hessian diagonal contribution of `b, m` against `α, g`. If `b, m` Fisher information is dominated by stimulus-side parameters by 1-2 orders of magnitude, recovery dynamics are weakly identified — informs successor X-channel additions (HRV, RHR, sleep).

#### Parameter-count sanity report

`(free shared parameters, per-athlete Z_0 parameters, observation rows, workout rows, rest rows, latent DOF, per-athlete rest-density)`. Per-athlete rest-density (C3) flags athletes with < 5% rest rows whose contribution to `b` identification is negligible. Reported per cohort and per athlete.

**Dispatch:** focused-engineer for diagnostic implementation; senior-scientist for interpretation before M6 closes.

---

## Open decisions

| # | Decision | Needed before | Status |
|---|---|---|---|
| W-D1 | Slug for the M6 estimator spec | W1 spec-landing | ⏸ defer to spec author |
| W-D2 | PyTorch vs JAX for differentiable Kalman | W1 draft | ⏸ open — spec author proposes; W3 ADR closes |
| W-D3 | Optimizer choice and learning rate | W1 draft | ⏸ open — spec author proposes |
| W-D4 | Multi-start seed count (3 / 5 / more) | W1 draft | ⏸ open — spec author proposes |
| W-D5 | Convergence criterion specifics | W1 draft | ⏸ open — spec author proposes |
| W-D6 | Validation-cohort `infer` convergence specifics | W1 draft | ⏸ open — spec author proposes |
| W-D7 | Init constants (`Q, Q_1` magnitudes; `Z_0` prior covariance) | W1 draft | ⏸ open — spec author proposes |
| W-D8 | `mode="smooth"` supported at M6 or deferred | W1 draft | ⏸ open — spec author proposes |
| W-D9 | Numerical precision (float32 vs float64) | W1 draft | ⏸ open — master-statistician closes |
| W-D10 | Rest-bound re-entry prior | W1 draft | ⏸ open — spec author proposes |
| W-D11 | Tier-1 tolerances from K = 20 replicate-variance study; per-parameter normalization scheme | W1 draft | ⏸ open — spec author proposes; master-statistician closes |
| W-D13 | Sign-convention cascade tolerances (`tol_b`, `tol_m`) | W1 draft | ⏸ open — spec author proposes |
| W-D14 | `ZPosterior` ABC interface (`mean()`, `sample(n)`, `marginal_log_pdf(z)` minimum) | W1 draft | ⏸ open — spec author proposes |
| W-D15 | Between-athlete equilibrium variance threshold for successor-1 trigger (provisionally 3×) | W1 draft | ⏸ open — spec author proposes |
| W-D12 | Smoothing (RTS pass) at fit time | W1 draft | ⏸ open — spec author proposes |

Dissolved by current draft: bootstrap procedures; Procrustes alignment; PCA-as-gauge; M4 fixable-`A` accommodation; rest-day plug-in rule; ADR 0004 Riegel-window seam (under SGD, the within-window self-normalization is a property of training data, not a leakage seam); free `τ` per dimension (now fixed schedule); free `H` (now fully diagonal); free `r_1` (now derived); covariance throwaway (now widened Protocol); `γ_j` / `σ_ij` `G`-parameterization redundancy (round 6 U2: now `g_ij` directly).

---

## Verification

| W | Verification |
|---|---|
| W1 | Spec passes README structure; commit-readiness clean. |
| W2 | No auditor returns a unanimous ask in their most recent pass; cross-auditor re-audit on material redrafts; audits logged. |
| W3 | ADR exists; cross-links to spec; concrete successor named (hierarchical `Z_0^{(i)}` + per-athlete `b^{(i)}`); EM-vs-SGD justification recorded; late-naming sentence; tier split with regime-shift Tier-C and quarter-scale-ceiling Tier-C; commit-readiness clean. |
| W4 | `pytest tests/ -v` passes; new tests cover staged init determinism, SGD determinism, `infer` returning `ZPosterior` with correct shapes for both cohorts, Prior.diffuse, d_Z mismatch, rest-bound, PSD preservation, multi-start, sign-convention idempotence, validation-cohort `Z_0` recovery on synthetic. |
| W5 | `pytest tests/ -v` passes; harness body tests including `run_evaluation` consuming `ZPosterior`. |
| W6 | Tier 1 passes at empirical replicate-variance tolerances (K=20 replicate study); Tier 2 numbers recorded; parameter-count report produced; senior-scientist interprets. Tier 1 failure re-opens W4. Tier 2 numbers never re-open at M6 (between-athlete-equilibrium-variance Tier-2 may fire successor 1 before M10 if the qualitative threshold is exceeded). |

Aggregate exit: full pytest passes; CLAUDE.md index current; sequencing plan updated; ADR cross-linked.

---

## Out of scope

- **Variational, hierarchical, non-Gaussian, or particle-filter estimators.** Successor reference impls.
- **Joint estimation of M4/M5's `fit` methods at M6.** M6 does its own joint fit; M4/M5's standalone `fit` not invoked.
- **Free `τ` time constants.** Fixed schedule at M6; relaxation is a successor choice.
- **Per-athlete `b^{(i)}`.** Cohort-shared at M6; per-athlete is the named successor direction (S2).
- **Time-in-zone / intensity-distribution stimulus channel.** Sports-science flagged as most important missing component; deferred to successor.
- **Non-diagonal `H`.** Rest is passive decay per dimension; cross-dimension rest coupling is a successor relaxation if data ever supports it.
- **`forward.py` / `predict.py` body implementations.** M7 (M6 lands the `ZPosterior`-consuming signatures only).
- **`evaluation/metrics.py`.** M8.
- **Selection model `p(P|Z)`.** Deferred per architecture_map §5.1.
- **Health-state projection `h(Z_t)`.** Not on prediction path.
- **Real-data cohort (M1b).** M6 validates on synthetic only.
- **Project-wide channel-assignment commitments.** Defers to M9/M10.
- **PyTorch/JAX dependency in non-`filter.py` modules at M6.** ADR accepts the dependency for `filter.py` only at this milestone.

---

## Deferred to M7+

- `forward_state` propagation of `ZPosterior` through a `ForwardSchedule`.
- `predict_session` composition consuming `ZPosterior` for calibrated prediction intervals.
- Reference-template projection (`project_to_reference`).

---

## Maintenance

- Workstream completions move from Active → Completed with commit SHA.
- Audits land under `### Audits landed` with date + one-line finding.
- W-Dn decisions resolve in Open decisions table.
- Scope drifts update Out of scope before code lands.
- Plan edits are their own commits.
- On M6 close: flip `Status` to `complete`; update `post-scaffold-sequencing.md` M6 row; record successor milestone (hierarchical `Z_0^{(i)}` + per-athlete `b^{(i)}`); add literature-anchor paragraph to ADR 0006 *Context* per the senior-scientist assessment (Banister/Busso/Hellard + Pinheiro-Bates NLME).
