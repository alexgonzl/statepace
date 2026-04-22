# Identifiability Baseline — Minimal Scalar (CCT-style) Model

Scope: the simplest observation model the architecture in
`docs/architecture_map.md` must support. One scalar observable per
session (cardiac-cost-type: an `X` channel per session summarizing
execution intensity relative to the session frame). Per-athlete fit.
Audit target: what is identified, what is assumed, what is at risk,
what restrictions close the minimum model. No estimator is proposed;
no architectural change is recommended.

Framework references: `docs/theoretical_framework.md` §§5–7; §A4, §A5,
§A7, §A8, §A10. Architecture: `docs/architecture_map.md` §3.2 (`ObservationModel`),
§3.3 (`WorkoutTransition`, `RestTransition`), §3.4 (`Prior`), §5 item 5
(dual-role flag).

## 1. Minimal family specification (audited, not proposed)

- `Z_t ∈ ℝ` (scalar, `d_Z = 1`). One capacity axis.
- `P_t`: session shape (distance, terrain). Observed; enters observation
  model, not transitions. No free parameters on `P`'s distribution
  (selection model deferred — §5.1 of architecture).
- `E_t`: exogenous (weather, TOD). Enters observation model linearly.
- `X_t ∈ ℝ`: scalar execution summary (e.g., cardiac cost).
- **Observation**: `X_t = μ(Z_{t-1}, P_t, E_t) + ε_X`,
  `ε_X ~ N(0, σ_X²)`.
- **Workout transition** `f`: `Z_t = Z_{t-1} + α·(X_t − x̄(P_t, E_t)) + ε_f`,
  `ε_f ~ N(0, σ_f²)`. Scalar training-stimulus gain `α`.
- **Rest transition** `g`: `Z_t = Z_{t-1} + β·(Z* − Z_{t-1}) + ε_g` for
  `n_rest ∈ [1, 10]`. Scalar mean-reversion rate `β`, asymptote `Z*`,
  `ε_g ~ N(0, σ_g²)`.
- **Prior** (`Prior` dataclass, §A8): `Z_0 ~ N(m_0, v_0)` with
  `v_0 → ∞`, `diffuse=True`. 1-year warm-up masked at harness.

This is the smallest model the architecture's Protocols can carry.

## 2. Parameter accounting

| Parameter | Role | Identification source | Status |
|---|---|---|---|
| `μ(·)` coefficients on `Z_{t-1}` | direct `Z → X` slope | Variation in `Z_{t-1}` at matched `(P, E)` over time within athlete | Identified conditional on enough `(P, E)`-matched days and sufficient `Z` trajectory variation |
| `μ(·)` coefficients on `P_t` | session-shape effect on `X` | Variation in `P_t` at matched `Z_{t-1}` (i.e., within a short window where `Z` is approximately constant) and matched `E` | Identified |
| `μ(·)` coefficients on `E_t` | exogenous effect on `X` | Variation in `E_t` at matched `(Z_{t-1}, P_t)` (A2 exogeneity is load-bearing) | Identified |
| `σ_X²` | observation noise | Residual variance after `μ(·)` fit on workout days | Identified |
| `α` (workout gain in `f`) | `X → Z` effect | Longitudinal change in `Z_{t-1}` regressed on prior `(X − x̄)`; requires temporal variation in `X_t` (§4d) | **Entangled with `μ`'s `Z`-slope** — see §3 |
| `σ_f²` | workout transition noise | Residual `Z`-increment variance on workout days | Identified conditional on `α` identified |
| `β`, `Z*` (rest transition) | recovery rate + asymptote | Rest-only sub-sample (`is_rest=True`, `n_rest ≤ 10`), via `Z`-change across pure rest runs | Identified on rest sub-sample; `Z*` is weakly identified per athlete (see §3) |
| `σ_g²` | rest transition noise | Residual `Z`-increment on rest days | Identified conditional on `β`, `Z*` |
| `m_0`, `v_0` (prior) | `Z_0` initial | Not identified from data (A8 diffuse) | **Assumed**, not identified |
| Scale/sign of `Z` | gauge | Nothing in the data | **Assumed** (convention: `α > 0`, and either `Var(Z)=1` or a fixed coefficient in `μ`) |

Selection model `p(P | Z)`: **not parameterized in the minimal model**,
consistent with architecture §5.1. Its absence is a modeling choice
with a cost, quantified in §4 concern 3.

## 3. Where the dual-role `X → Z` edge bites

Mechanism. On any workout day, the same scalar `X_t`:

- Appears in the observation likelihood
  `log p(X_t | Z_{t-1}, P_t, E_t)` through `μ(Z_{t-1}, P_t, E_t)` — the
  data constrain the slope `∂μ/∂Z`.
- Appears in the transition likelihood
  `log p(Z_t | Z_{t-1}, X_t)` through `α·(X_t − x̄)` — the data
  constrain the gain `α`.

Both channels are only queryable at the *same* `X_t` realization; there
is no counterfactual `X_t` given `Z_{t-1}` (§4b). Consequence:

1. **Joint workout-day likelihood identifies a product.** Rescaling
   `Z → c·Z` with `∂μ/∂Z → (∂μ/∂Z)/c` and `α → α·c` is a likelihood-
   invariant reparameterization. A gauge restriction (fix `Z`'s scale,
   or fix one coefficient in `μ`) is mandatory.
2. **Slow `Z` drift + noisy `X` reduces `α` to a residual-on-residual
   coefficient.** When `Var(Z_{t-1})` is small relative to `σ_X² + σ_f²`,
   the observation model cannot separate `μ`'s `Z`-slope from the
   `X_t − x̄` residual that drives `Z_t`; `α` and `∂μ/∂Z` exchange.
3. **`Z*` in `g` gets its only *cross-calibration* through `f`.** Rest
   days identify `β` and `Z*` on the *Z-scale*, but that scale is set by
   `f`. If `α` is weakly identified, `Z*` drifts with it — even though
   the rest sub-sample is clean.
4. **Constant-`X` regimes (§4d) are silent on `α` and on `∂μ/∂Z`
   simultaneously.** The two edges fail together, not independently,
   because `X` is the shared input.

What you can estimate from workout days alone in the minimal model:
- The *effect of `(P, E)` on `X`* at a given (possibly time-varying)
  capacity — identified up to the `Z`-gauge.
- The *composite* `(∂μ/∂Z, α)` — identified *only* as a 1-d curve in the
  2-d parameter space without a gauge restriction, and weakly identified
  even with one when the `Z` trajectory is smooth.

## 4. Concerns, ranked by severity

1. **Dual-role parameterization of `X_t` is not forced coherent by the
   architecture (scientist ask (b)).** `observation.log_prob` and
   `WorkoutTransition.step` receive the *same* `X` object (architecture
   §5 item 5), but the architecture does not constrain *how `X`
   parameterizes each side's effect*. Two independent functional
   choices (e.g., `μ(Z, P, E) = θ_μ·g_μ(X_t-components,Z,P,E)` in the
   observation, `α·X_t` in `f`) give an estimator free rein to absorb
   the training-stimulus into the observation intercept. **Consequence**:
   the estimator converges to a joint minimum that is a mixture of
   observation drift and training effect; neither is interpretable.
   **Class of restrictions that closes it** (family-level, not
   architecture-level): (i) **shared coefficient / shared basis** — the
   per-component `X`-effect in `μ` and `f` is the same functional form
   up to a known link; (ii) **monotone-link coupling** — `f`'s drive is
   a monotone function of `μ`'s residual `X − μ(Z, P, E)` rather than of
   raw `X`, so the observation model cannot absorb training effect
   without enlarging its residual; (iii) **residual-only driver** — `f`
   is driven exclusively by the observation residual (what the body did
   *beyond what capacity predicted*). Each is a different scientific
   commitment; picking one is outside this audit's scope.
2. **Gauge on `Z`.** Without fixing `Z`'s scale and sign, the full
   workout-day likelihood has a one-parameter continuous symmetry
   (`α ↔ ∂μ/∂Z`). **Estimator effect**: non-identifiable; ridge or
   weak prior will pick *a* point on the curve. **Fix**: fix scale
   (e.g., `Var(Z_fit_window) = 1` or one unit coefficient in `μ`) and
   sign (`α > 0`). Required.
3. **Selection-ignorability (scientist ask (a)).** The architecture
   defers `p(P | Z)`; the baseline invokes **selection on observables**:
   estimation conditions on the realized `P_t`. Under A2 (E-exogeneity)
   and the DAG, `Z_{t-1} → P_t → X_t` is a mediator path, not a
   backdoor. So conditioning on `P_t` *does* block the mediated effect
   and *does not* open a backdoor. **What that costs**: it identifies the
   **direct-effect** observation coefficient — `E[X | Z, P, E]` with `P`
   fixed — which is the right target for filtering and for prediction
   at an observed future `P_{t+τ}` (§7). It is **not** the right target
   for counterfactual capacity ("how fit would this athlete be if they
   had trained differently"), because that query requires intervening
   on `Z_{t-1}`'s *downstream* `P` channel, i.e., the selection model.
   **Consequence of deferring `p(P | Z)`**: filtering and one-step
   prediction at realized `P` are valid; total-effect / counterfactual
   statements are out of scope. `ConditioningSpec.p_mode="marginalized"`
   is unreachable in the minimal model, exactly as the architecture
   (§5.1) already enforces.
4. **`d_Z = 1` vs §8's high-dimensional claim.** The framework commits
   to `Z` carrying current fitness, acute fatigue, stable traits,
   periodization regime, aging drift (§8). The minimal scalar `Z`
   collapses all of this onto one axis. **Consequence**: the scalar `Z`
   trajectory is forced to absorb aging, tapers, and post-injury
   shifts as non-stationary drift on a single dimension; under A10
   stationarity of `f`, `g`, `μ`, this looks like mis-specification
   rather than a valid trajectory. **What is identified anyway**: a
   single "readiness" axis that tracks whatever scalar the observation
   model is most informative about. Interpretation limited; prediction
   defensible within the warm-up-clean window.
5. **Temporal variation in `X` (§4d).** The minimal model's `α` is
   identified only if `X_t` varies across workout days at non-trivial
   amplitude relative to `σ_X² + σ_f²`. Flat training blocks silently
   un-identify `α`. **Fix**: diagnostic on `Var(X_t)` within-athlete;
   defer to the estimator to flag.
6. **Rest sub-sample may be small.** Per athlete, `g`'s `β, Z*` are
   identified on pure-rest-day `Z`-change residuals within 10-day
   windows. If the athlete rarely rests, or rests only in gaps >10
   days (A5 out-of-scope), `g` is weakly identified and leaks into
   `f`'s estimation via `Z_{t-1}` passed across rest days.
7. **Warm-up leakage.** Conventions require a 1-year warm-up mask at
   the harness (architecture §3.7, `warmup_days=365`). The estimator
   itself is unaware. **Risk**: early-trajectory `Z_{0:365}` carries
   diffuse-prior residue; if a caller bypasses `EvalSplit`, prior bias
   leaks. **Fix**: harness-level enforcement is the existing defence;
   callers must not bypass.
8. **Stationarity (A10) vs regime shifts.** Injury / layoff / coach
   change violates A10 within-athlete. Minimal model has no switching
   mechanism. **Consequence**: `f`, `g`, `μ` estimates are a weighted
   average across regimes; predictions at regime boundaries are biased.
   **Fix**: out of scope for the baseline; flag in fitting.
9. **Re-entry after `n_rest > 10`.** The architecture (§5 item 2)
   defers the re-entry-policy choice. The minimal model as written has
   no mechanism; `Z` is undefined post-gap. Practical identification
   consequence: post-gap `Z` initialization is another unidentified
   choice that silently influences downstream `f` estimates until the
   trajectory re-stabilizes.

## 5. Minimum restrictions for the minimal model to be identified

All five are necessary; none is sufficient alone.

1. **Gauge on `Z`.** Fix scale and sign. Concretely: either constrain
   `Var(Z over fit window) = 1` or set one coefficient in `μ` (e.g.,
   the coefficient on a standardized `Z`-slope component) to 1; and
   sign-constrain `α > 0`. Closes concern 2.
2. **Dual-role coherence restriction.** Commit to one of: shared
   coefficient across `μ` and `f`; monotone link between them; or
   residual-only driver in `f` (`f` sees `X_t − μ(Z_{t-1}, P_t, E_t)`,
   not raw `X_t`). Closes concern 1. Selection of *which* is a family
   decision, not an identifiability one; the audit flags the class.
3. **Linear / additive functional form.** Observation `μ` linear
   in `(Z, P, E)`; `f` additive in a scalar function of `X`; `g`
   linear mean-reversion. Nonparametric `μ`, `f`, `g` on a scalar
   `Z` with shared workout-day data is under-identified even with
   gauge + dual-role restrictions, because flexible `μ` absorbs
   flexible `f`.
4. **Drop `m_0, v_0` from the estimated set.** `Prior.diffuse=True`
   (explicitly, via the `Prior` dataclass) and 1-year warm-up masking
   at the harness (conventions §warm-up). No population prior
   estimated in the minimal (per-athlete) model.
5. **Restrict scope to filtering and one-step prediction at realized
   `(P, E)`.** Defer `p(P | Z)`; forbid `p_mode="marginalized"` in
   any `ConditioningSpec` the minimal model services. Closes concern
   3's scope at the cost of counterfactual claims — a cost the
   minimal model must accept.

## 6. What is still not identified under all five restrictions

- **Selection / total-effect parameters.** By construction, deferred.
- **`Z_0` numerics.** Diffuse-by-assumption.
- **Regime-shift breakpoints.** A10 stationarity assumed.
- **Re-entry degradation.** Architecture defers.

These are scoped out, not failures. Scoring gates at 1-year warm-up
and to `n_rest ≤ 10` windows keep them off the score.

## 7. Verdict on the minimal model as an identifiability *baseline*

Under restrictions 1–5: **partially identified**, sufficient for
filtering and one-step prediction at realized `(P, E)`, insufficient
for counterfactual capacity claims. This is the correct baseline — it
is the minimum at which the architecture's current Protocols
(`ObservationModel`, `WorkoutTransition`, `RestTransition`, `Prior`)
carry a model whose parameters all map to a data regime.

Without those five restrictions: **under-identified**, dominated by the
dual-role coupling of `X_t` across `observation.log_prob` and
`WorkoutTransition.step` and by the `Z`-gauge symmetry.
