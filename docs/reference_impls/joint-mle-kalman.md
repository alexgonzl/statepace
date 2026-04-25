# Joint MLE via differentiable Kalman filter (`joint-mle-kalman`)

**Protocol:** `StateEstimator`
**Status:** draft

## Summary

Reference `StateEstimator` for `statepace`. Joint maximum-likelihood fit of all shared parameters and per-athlete `Z_0^{(i)}` by SGD on the cohort negative log-likelihood, with a differentiable Kalman filter integrating over the latent `Z` trajectory. Structural priors (multi-timescale `Z`, fixed `τ`, diagonal `H`, cohort-shared detraining target, headroom-bounded stimulus loading) are baked into the parameterization and cannot be escaped by the optimizer. Constrained-MLE first reference; successor families (see `docs/plans/m6-state-estimator.md`) relax one prior at a time.

Plan: [`docs/plans/m6-state-estimator.md`](../plans/m6-state-estimator.md).

## Channel composition

Paired at bundle-construction time with [`riegel-score-hrstep`](riegel-score-hrstep.md) (M4) and [`linear-gaussian`](linear-gaussian.md) (M5). The estimator does not call M4's or M5's `fit`; it consumes their `log_prob` likelihoods inside the joint Kalman filter. P, E, X, `π_obs`, `π_stim` definitions are inherited from the paired specs without modification. No new channel commitments here.

## Family

### Joint state-space form

Per athlete `i`, with `t` over training days:

- **Workout day** (M4 emission + M5 `f`):
  ```
  X̃_t = A·Z_t + B·P_t + C·E_t + d + ε_t,        ε_t ~ N(0, Σ)
  Z_t = F·Z_{t-1} + G·(π_stim(X_t) − π̄_stim) + m + ν_t,   ν_t ~ N(0, Q)
  ```
- **Rest day** (M5 `g`, no observation update):
  ```
  Z_t = H·Z_{t-1} + r_1 + ν_t,                  ν_t ~ N(0, Q_1)
  ```
- **Initial state:** `Z_0^{(i)} ~ N(μ_0^{(i)}, Σ_0)` with `Σ_0` diffuse (§A8); `μ_0^{(i)}` is a per-athlete fit parameter.

Multi-step rest is the M5 single-step recursion iterated; no separate `n`-day form.

### Structural parameterization

Dimensions referred to as `Z_1, Z_2, Z_3, Z_4` indexed by `τ_i`. Physiological correspondence is suggestive, not committed (CLAUDE.md late-naming).

- **Latent dimensionality:** `d_Z = 4`.
- **Time constants (fixed):** `τ = (τ_1, τ_2, τ_3, τ_4) = (1, 7, 28, 84)` days. Symmetric across `F` and `H`.
- **`F` (workout AR):**
  ```
  F_ii = exp(−1/τ_i)                       (fixed)
  F_ij = α_ij · (1 − F_ii)   for i ≠ j     (free via β_ij)
  α_ij = tanh(β_ij)                        (β_ij ∈ ℝ free)
  ```
- **`H` (rest decay):** fully diagonal,
  ```
  H_ii = exp(−1/τ_i)                       (fixed)
  H_ij = 0     for i ≠ j                   (no rest cross-coupling)
  ```
  Automatically satisfies M5's `0 < λ(H) < 1`.
- **`b ∈ ℝ^{d_Z}`:** cohort-shared detraining target, free.
- **`r_1`:** derived, not fit. `r_1 = (I − H) · b`.
- **`m ∈ ℝ^{d_Z}`:** workout-day drift, free (cohort-shared).
- **`G` (stimulus loading):**
  ```
  G_ij = (1 − F_ii) · g_ij                 (g_ij ∈ ℝ free)
  ```
  Sign unrestricted. Per-row bound `(1 − F_ii)` keeps the slow rows bounded against arbitrary stimulus-driven excursions.
- **`Σ, Q, Q_1`:** Cholesky-parameterized PSD. `Σ = L_Σ L_Σᵀ`, `Q = L_Q L_Qᵀ`, `Q_1 = L_{Q_1} L_{Q_1}ᵀ`. Diagonal elements via softplus.
- **`B, C, d`:** unconstrained (M4-side regression coefficients).
- **`A`:** unconstrained `(d_X, d_Z)` at fit time; PCA-initialized (see §Initialization).
- **`Z_0^{(i)} = μ_0^{(i)} ∈ ℝ^{d_Z}`:** per-athlete, free; `Σ_0` fixed diffuse (W-D7 magnitude).
- **Rest-bound re-entry prior:** `Z ~ N(b, σ_re² · I)`. Mean fixed to the cohort detraining target on the rationale that `b` is the absorbing equilibrium of `g`. Variance `σ_re²` declared at W-D10. Applied at the first post-gap workout-day observation when a rest run exceeds `max_consecutive_rest_days`.

### `π_stim` centering (cohort-mean offset)

Compute `π̄_stim` as the cohort-mean of `π_stim(X_t)` over all training-cohort workout days. Subtract from `π_stim` everywhere it is consumed (training fit, validation `infer`, all subsequent `infer` calls). `π̄_stim` is **frozen at training time** and re-applied unchanged downstream; recomputing on a different cohort would shift `m`'s intercept basis and bias `Z_0^{(i)}` recovery.

The shift is absorbed into `m`'s init; it dissolves the practical `m ↔ G·E[π_stim]` trade-off without adding a parameter.

### Loss

```
L(θ, {Z_0^{(i)}})  =  − Σ_i log p(X^{(i)} | θ, Z_0^{(i)})
```

where `log p(X^{(i)} | θ, Z_0^{(i)})` is the marginal log-likelihood under the joint state-space model, computed by a **differentiable Kalman filter** with `Z` integrated out analytically (linear-Gaussian → Kalman is exact). Per-day contributions:

- **Workout day:** Kalman predict (under `f`) + update (under M4 emission).
- **Rest day:** Kalman predict only (under `g`).
- **Rest-bound overrun:** observation rows beyond `max_consecutive_rest_days` of consecutive rest are masked from the loss; the filter is reset at the first post-gap workout day under a re-entry prior `Z ~ N(b, Σ_re)` (re-entry mean = cohort detraining target; `Σ_re` declared, W-D10).
- **Warm-up days** (first 6 months per athlete, §A8): rows are scoring-masked at evaluation; at fit time they enter the loss (the diffuse `Z_0` prior is the A8 defence).

### Optimizer

- **Framework:** PyTorch (proposal; W-D2). Differentiable Kalman written in PyTorch with batched matrix ops. JAX is the alternative; PM arbitrates at W3 ADR.
- **Optimizer:** Adam (proposal; W-D3). LR `1e-3` initial, cosine decay to `1e-5` over `max_iterations`.
- **Convergence:** stop when validation NLL has not improved by `> tol_conv` (W-D5; proposal `1e-4` relative) for `patience` consecutive evaluations, or `max_iterations` reached. Validation NLL evaluated every `eval_every` epochs on a held-out cohort split.
- **Multi-start:** `n_seeds = 5` (proposal; W-D4). Each seed re-randomizes the small-random init components (β, b, m, g, Cholesky factors); deterministic-init components (`A` from PCA, `B/C/d` from OLS) are seed-independent. **Selection rule:** highest validation log-likelihood.
- **Numerical precision:** float64 (W-D9 closed in spec). The Kalman pass conditions on the Cholesky of state covariance; float32 produces PSD drift on long sequences with small `Q_1`.
- **PSD hygiene:** Cholesky-parameterized covariances are PSD by construction. Joseph-form Kalman update used if vanilla form drifts.
- **Smoothing:** `mode="filter"` only at M6 (W-D8 closed in spec — defer RTS smoother). `infer(mode="smooth")` raises `NotImplementedError` until a successor adds it.

### Initialization (staged)

Deterministic given the training cohort, except small-random components which are re-seeded per multi-start.

| Parameter | Init |
|---|---|
| `A` | PCA of warmup `X̃` pooled across training cohort, top `d_Z = 4` directions |
| `B, C, d` | OLS of `X̃` on `(P, E, 1)` over warmup |
| `Σ` | Residual covariance from the `(P, E)` regression |
| `β_ij` | 0 (so `α_ij = 0` and `F` starts diagonal) |
| `b` | small random near zero (per seed) |
| `m` | small random near zero (per seed); cohort-mean offset already absorbed via `π_stim` centering |
| `g_ij` | small random near zero (per seed) |
| `Q, Q_1` | `q_init · I`, `q_init` small (W-D7; proposal `0.01`) |
| `μ_0^{(i)}` | zero vector |
| `Σ_0` (fixed) | `σ_0² · I`, `σ_0²` large (W-D7; proposal `100`) |

### Validation-cohort `infer`

For an athlete not in the training cohort:

1. Freeze `θ` at the training-fit values (including `π̄_stim`).
2. Fit `μ_0^{(i)}` (and only `μ_0^{(i)}`) by SGD on the single-athlete NLL under the frozen `θ`. Same code path as training-fit `infer`. Adam, same LR schedule, single-seed at `infer` time. Identifiability of `μ_0^{(i)}` under frozen `θ` is W2 audit territory (NA1 recovery test in W6 is the empirical check).
3. Run the Kalman filter forward with the fitted `μ_0^{(i)}`.
4. Return `GaussianZPosterior(mean, cov, dates)`.

`Prior.diffuse=True` is honored via the same large `Σ_0`.

### Post-hoc gauge convention (sign cascade)

Fixed `τ` schedule + diagonal `H` pin diagonal-ordering and rotation gauges. Residual gauge: per-dimension sign flips (`Z_i → −Z_i` with simultaneous flip of `A` row `i`, `b_i`, `m_i`, `G_{i·}`, `(L_Q)_{i·}`, `(L_{Q_1})_{i·}`) are likelihood-invariant.

For each dimension `i = 1..d_Z`, declare positive sign by cascade:

1. **Primary:** `b_i ≥ 0`. Apply if `|b_i| ≥ tol_b` (W-D13; proposal `tol_b = 0.05` in standardized units).
2. **Fallback:** `m_i ≥ 0`. Apply if `|b_i| < tol_b` and `|m_i| ≥ tol_m` (W-D13; proposal `tol_m = 0.05`).
3. **Final fallback:** signed sum of `G_{i·}` ≥ 0. Apply if both primary and fallback below tolerance.
4. **Boundary flag:** if all three tie-breakers fall below their tolerances, leave dimension `i` unflipped and emit a flag in the fit artifact (`gauge_boundary_dims`).

Applied **post-fit, for diagnostics only.** Imposes no constraint on the optimizer. Idempotent: re-applying after a flip is a no-op. Boundary flags propagate to the M6 fit artifact; downstream policy (M7's `predict.py`) is deferred.

### Fit artifact

The fitted estimator carries:

- All `θ` blocks above.
- Per-athlete `μ_0^{(i)}` for the training cohort.
- `π̄_stim` (frozen).
- `gauge_boundary_dims`: list of dimensions flagged at the gauge boundary.
- `selection_log_lik`: validation log-likelihood of the selected seed.
- `multi_start_diagnostics`: per-seed final loss + selected seed index.

### `infer` Protocol widening

```
StateEstimator.infer(channels, mode="filter", prior=None) -> ZPosterior
```

`ZPosterior` is a **sealed ABC** (W-D14) with minimal interface:

```
class ZPosterior(ABC):
    dates: Array
    d_Z: int
    @abstractmethod
    def mean(self) -> Array: ...                                  # (T, d_Z)
    @abstractmethod
    def sample(self, n: int, rng) -> Array: ...                   # (n, T, d_Z)
    @abstractmethod
    def marginal_log_pdf(self, z: Array) -> Array: ...            # (T,) at given z trajectory
```

M6 returns `GaussianZPosterior(mean: (T, d_Z), cov: (T, d_Z, d_Z), dates)` implementing the ABC. Successor families (variational, particle, switching) define their own concrete subclasses without re-widening the Protocol. M6's W4 lands the ABC + `GaussianZPosterior` in scaffold; `forward.py` and `predict.py` consume the ABC interface starting M7.

### Hyperparameters surfaced by name

| Name | Description | Default / W-D |
|---|---|---|
| `d_Z` | Latent dimensionality (matched across paired impls) | 4 |
| `max_consecutive_rest_days` | Rest-bound (M5 §A5) | matched to M5 |
| `learning_rate` | Adam initial LR | `1e-3` (W-D3) |
| `max_iterations` | Iteration cap | declared per ADR (W-D3) |
| `patience` | Validation-NLL plateau patience | declared (W-D5) |
| `eval_every` | Validation-NLL eval cadence (epochs) | declared (W-D5) |
| `tol_conv` | Validation-NLL relative-improvement tolerance | `1e-4` (W-D5) |
| `n_seeds` | Multi-start seed count | 5 (W-D4) |
| `q_init` | Init magnitude for `Q, Q_1` diagonals | `0.01` (W-D7) |
| `sigma0_sq` | Diffuse-prior variance for `Z_0` | `100` (W-D7) |
| `tol_b` | Sign-cascade primary tolerance | `0.05` (W-D13) |
| `tol_m` | Sign-cascade fallback tolerance | `0.05` (W-D13) |
| `sigma_re_sq` | Rest-bound re-entry prior variance | declared (W-D10) |

## Missingness

- **Rest days** (`X_t.is_rest = True`): no observation update; Kalman predicts under `g` (M5 rest transition). Likelihood contribution is the rest-day marginal `p(Z_t | Z_{t-1})` integrated by the filter.
- **No-valid-effort days** (workout day with no qualifying best effort, per M4): treated as rest at the observation step (no `π_obs` row), but `π_stim` aggregates remain available, so `f` (workout transition) still applies. Equivalent to a rest-style update in the observation channel and a workout-style update in the transition channel for that day.
- **Warm-up mask:** first 6 months of each athlete's history (§A8) are not score-masked at fit time — they enter the joint NLL — but are score-masked at evaluation. The diffuse `Z_0` prior is the A8 defence at fit time.
- **Rest-bound overrun:** consecutive rest stretches beyond `max_consecutive_rest_days` are masked from the loss for those rows; filter resumes at the first post-gap workout-day observation under the re-entry prior `Z ~ N(b, σ_re² · I)` (`b` is the fitted detraining target; `σ_re_sq` declared at W-D10). The harness reports rest-bound violations via `EvalResult.rest_bound_violations` (M5 boundary rule).

## Coherence with other impls

What M6 commits to about M4 and M5:

- **Does not call `M4.fit` or `M5.fit`.** M4 and M5 expose `fit` for unit-testing isolation; the joint estimator subsumes them by consuming `M4.log_prob` and `M5.log_prob` inside the differentiable Kalman filter. The joint NLL is the sum of M4's observation likelihood + M5's transition likelihoods (per ADR 0005's edge assignment).
- **Inherits M4's pre-transforms `X → X̃`.** The Kalman update operates on `X̃`. M4's `forward` and `inverse` invert the transform on output; M6's `infer` consumes `X̃` internally and produces `ZPosterior` in `Z`-space.
- **Inherits M5's `(F, G, H, m, b, r_1, Q, Q_1)` family but pins their parameterization.** M5's spec leaves `H = I − A` with `A` symmetric PSD and bounded; M6 narrows to fully diagonal `H_ii = exp(−1/τ_i)`. M5's spec leaves `r_1` free; M6 derives it from `r_1 = (I − H)·b`. These narrowings are M6's structural priors and do not mutate M5's spec.
- **`d_Z` matching:** estimator's `d_Z` equals paired impls' `d_Z`. Construction-time error on mismatch.
- **`π_stim` cardinality:** matches M5's input shape (5 components under the `riegel-score-hrstep` pairing); `G` shape `(d_Z, 5)`.
- **`π̄_stim` is M6-side state.** M4 and M5 do not see the centering offset; it is applied inside M6's loss assembly before passing `π_stim` to M5's `log_prob`.
- **`infer` Protocol widening lands at M6 W4.** Scaffold updates in `statepace/filter.py` (return type), `statepace/forward.py` and `statepace/predict.py` (signatures consume `ZPosterior`).

## What would invalidate this

Tier A: re-open W1 (spec/contract bug). Tier B: re-open W3 (ADR-level miscalibration). Tier C: close M6 with finding; open named successor milestone (rows 1–6 of plan §W3 successor table).

### Tier A — contract / impl

- `d_Z` mismatch between estimator and paired impls not raised at construction.
- Rest-bound overrun rows leaking into the loss.
- PSD drift in `Σ, Q, Q_1, Σ_0|t` across SGD iterations (Joseph form not engaging when needed).
- Non-determinism under fixed seeds (multi-start not reproducible).
- Sign-convention cascade not idempotent or producing multi-start parameter divergence inconsistent with sign-flip alone.
- `π̄_stim` recomputed at validation `infer` instead of frozen at training time.

### Tier B — ADR-level miscalibration

- Staged init recipe insufficient: reliable non-convergence across seeds, or recovery on in-family synthetic systematically biased even at well-converged loss.
- Learning-rate schedule clearly miscalibrated (loss diverges, oscillates, or stalls before plateau across all seeds at default LR).
- Multi-start count too low: spread across `n_seeds = 5` exceeds empirical replicate-variance tolerance on parameters known to be non-weak (i.e., outside C1's slow-row off-diagonals and outside `gauge_boundary_dims`).
- PyTorch (or chosen framework) dependency adoption blocked; framework swap required.
- `mode="smooth"` demanded by downstream consumers before M6 closes (forces W-D8 reversal).

### Tier C — close M6 with finding; open successor milestone

Each row maps 1:1 to plan §W3 successor table.

| Tier-C condition | Successor row |
|---|---|
| Cohort-shared `(b, m)` mis-fits mixed-ability cohorts; between-athlete equilibrium variance ratio exceeds W-D15 threshold (provisional 3×). | 1 — hierarchical per-athlete `(b^{(i)}, m^{(i)})` |
| Cross-dimension rest coupling shows systematic sign in stratified rest-day residuals (post-fit diagnostic). | 2 — free off-diagonal `H` |
| Time-constant identifiability check fails: `τ` near `(1, 7, 28, 84)` rejected, or longer-window data identifies a `τ_4 > 84` mode. | 3 — free `τ_i` (and/or extended train window per row 6) |
| Heavy-tailed observation residuals; Gaussian likelihood mis-weights outliers detectably. | 4a — non-Gaussian observation likelihood |
| `GaussianZPosterior` interval calibration on M10 fails post-warmup coverage. | 4b — variational / non-Gaussian posterior |
| Regime-shift contamination: per-athlete `θ_i` vs shared-`θ` pseudo-likelihood ratio large for athletes with mid-window injuries / layoffs / periodization re-points (A10 violation). | 5 — switching SSM |
| `τ_4 = 84` insufficient for chronic adaptation on M10; train window does not identify longer modes. | 6 — extended train window |

**Logical closure:** M6 closes on clean Tier-1 acceptance diagnostics (W6) + clean Tier A. Tier B re-opens W3. Tier C fires retrospectively after M6 has closed, opening a named successor milestone.
