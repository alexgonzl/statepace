# Linear-Gaussian transitions (`linear-gaussian`)

**Protocol:** `WorkoutTransition` + `RestTransition`
**Status:** draft

## Summary

Reference `WorkoutTransition` + `RestTransition` pair for `statepace`. `f` (workout) is a linear-Gaussian autoregression in `Z` with a linear stimulus loading on `π_stim(X_t)`. `g` (rest) is a stationary linear-Gaussian recovery where the n-day transition is the one-day recovery matrix iterated n times, with derived drift and accumulated noise. `d_Z` is a sweep-time hyperparameter, matched to the paired observation impl's `d_Z` at bundle construction. Reference-impl scope only; framework commits to no privileged form (ADR 0002).

## Channel composition

This reference impl is paired with `riegel-score-hrstep.md` for its observation side. `f` reads `π_stim(X_t)` from the paired observation impl — the 5-component load vector declared there (`hr_load, step_load, total_elevation_gain, total_elevation_lost, heat_exposure`). `g` reads no component of `X`; rest days have no execution signal by construction. No `P`, `E`, or `π_obs` components enter `f` or `g` — per ADR 0005's `π_stim`-only stimulus rule.

## Family

### Workout transition (`f`)

```
Z_t = F · Z_{t-1} + G · π_stim(X_t) + m + ν,   ν ~ N(0, Q)
```

- `F`: `(d_Z, d_Z)` autoregressive matrix.
- `G`: `(d_Z, len(π_stim))` stimulus-loading matrix. Shape follows the paired observation impl's `π_stim` cardinality; `len(π_stim) = 5` under `riegel-score-hrstep`.
- `m`: `(d_Z,)` drift.
- `Q`: `(d_Z, d_Z)` PSD process-noise covariance.

### Rest transition (`g`)

Stationary recovery: the n-day transition is the one-day transition iterated n times.

```
Z_t = H^n · Z_{t-1} + r(n) + ν,   ν ~ N(0, Q_rest(n))
```

Free parameters are `H`, `r_1`, `Q_1` — the one-day transition quantities. `r(n)` and `Q_rest(n)` are derived functions of `(H, r_1, Q_1, n)`, with `n` supplied per call; they are not independently parameterized per n.

- `H`: `(d_Z, d_Z)` one-day recovery matrix, parameterized as `H = I − A` with `A` symmetric and `0 < λ(A) ≤ 1`. This keeps `H` PSD with eigenvalues in `[0, 1)`: monotone decay, no oscillatory modes, no unit roots, no explosive eigenvalues.
- `r_1`: `(d_Z,)` one-day drift. Derived n-day drift: `r(n) = (I − H^n)(I − H)⁻¹ r_1` (equivalently, the recursion `r(n) = H·r(n−1) + r_1` with `r(0) = 0`, which is the numerically stable form for implementation).
- `Q_1`: `(d_Z, d_Z)` PSD one-day noise covariance. Derived accumulated covariance: `Q_rest(n) = Σ_{k=0}^{n−1} H^k Q_1 (H^k)ᵀ` (equivalently, the recursion `Q_rest(n) = H·Q_rest(n−1)·Hᵀ + Q_1` with `Q_rest(0) = 0`).

`H` is identified from `n=1` rest-day data alone. The `H^n` iterated form extrapolates to `n ≥ 2`; residuals at `n ≥ 2` are held out as a mis-specification check, not used to identify `H`.

### Hyperparameters surfaced by name

- `d_Z` — latent dimensionality. Set at construction; matched across the observation and transition impls in a sweep bundle.
- `max_consecutive_rest_days` — validity bound for `g` (§A5). Beyond this bound, `Z_t` is undefined; callers must treat Z as having exited the state-tracked regime.

### Z-gauge

Z-gauge is a joint constraint across `A` (observation), `F`, `G`, `H` (transitions), and all covariances. This spec does not pin a gauge; the M6 estimator reference impl owns the joint gauge-fixing convention.

## Missingness

Workout days (`is_rest=False`) route to `f`. Rest days (`is_rest=True`) route to `g`, with `n_rest_days` set by the caller as the count of consecutive rest days ending at the current day (1 through `max_consecutive_rest_days`). Sequences of rest beyond the bound are out-of-contract for `g`; the caller must treat Z as undefined past the bound (A5).

## Coherence with other impls

- `π_stim` components must match the observation impl in a bundle. Paired with `riegel-score-hrstep`, the 5-component set is fixed; `G` is shaped accordingly.
- `d_Z` must match the observation impl at fit time; mismatch is a construction error.
- M5 Protocols (`step`, `log_prob`) take `Z` as input; whether callers pass posterior mean or full posterior distribution is an M6 estimator-side concern, deferred.
- ADR 0005 governs the single-node `X_t` plus per-edge functionals view; this spec instantiates `π_stim` consumption accordingly.

## What would invalidate this

- Flat `G` directions across athletes (sign-flipping rows at similar residuals): `π_stim` effective rank < 5, needs cohort-level rank reduction.
- Non-zero `cov(ν̂, ε̂)` after fitting: A9 leakage (omitted driver of both regimes).
- `Z`-residuals depend on lagged `hr_drift` (a π_obs component): ADR 0005's `π_obs`/`π_stim` split may need revisit for this family.
- Stratified n-rest-day residuals show systematic sign for n≥2: `H^n` iterated-matrix assumption wrong-shape.
- Rest-bound-boundary residuals monotone in n as n → `max_consecutive_rest_days`: either the bound is too loose, or `g` is bleeding into the detraining regime (A5 boundary violation).
- `§8` loads beyond fitness / fatigue / trait / trend: `d_Z` too small (expected under-parameterization at small `d_Z`).
- `F`/`G` collinearity via selection `P ← Z_{t-1}`: parameter-swap likelihood test flat at M6, indicating workout-day data identifies the combined `F·Z + G·π_stim` but not `F` and `G` separately.
- Asymmetric gain/loss in capacity response: linear `F` cannot represent faster loss than gain (or vice versa); residuals with sign-dependent structure flag this.
