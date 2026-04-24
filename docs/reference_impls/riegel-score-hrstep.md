# Riegel-score / hr+step load (`riegel-score-hrstep`)

**Protocol:** `ObservationModel`
**Status:** draft

## Summary

Reference `ObservationModel` for `statepace`. `π_obs` extracts the session's best effort by Riegel-relative speed (the continuous segment maximizing `observed_speed / riegel_speed(d)` over a valid distance range) and scores it against the per-athlete Riegel curve fit on the training window (ADR 0004). `π_stim` carries whole-session load aggregates: Edwards-TRIMP HR load, speed-weighted step load, total elevation gain/lost, and integrated heat exposure. Reference-impl scope only; framework commits to no privileged form (ADR 0002).

## Channel composition

Reference-impl scope. Does not commit the project to a channel assignment (CLAUDE.md standing rule defers project-wide channel calls to M9/M10).

- **P:** `total_distance, total_duration, elevation, is_track, time_of_day_sin, time_of_day_cos`.
- **E:** `wet_bulb_temp`.
- **X:** `best_effort_riegel_speed_score, best_effort_grade, best_effort_mean_HR, best_effort_hr_drift, best_effort_speed_cadence_ratio, hr_load, step_load, total_elevation_gain, total_elevation_lost, heat_exposure`.

`P.elevation` and `X.total_elevation_gain/lost` are distinct quantities, not different summaries of the same thing. `P.elevation` is the median altitude of the session — an ambient environmental frame (altitude physiology). `X.total_elevation_gain/lost` are integrated realized climb/descent over the session — execution quantities routed into `π_stim` because they drive adaptation via `f`. The route-vs-execution split: the athlete chooses *where* to run (P) and the body produces *what happened* (X).

## Functionals (per ADR 0005)

### `π_obs(X_t)` — observation projection

Best-effort extraction over the within-session trajectory. A "valid effort" is a continuous running segment of distance `d ∈ [d_min, d_max]`; the best effort is the segment whose observed mean speed most exceeds the athlete's Riegel-predicted speed at that segment's distance (i.e., maximizing `observed_speed / riegel_speed(d)`). `d_min` and `d_max` are hyperparameters surfaced by name, not fixed here. A session qualifies as observation-contributing iff it contains at least one continuous segment of length `≥ d_min`; QA filtering of the upstream session stream is assumed. Sessions with no valid effort do not produce a `π_obs` readout (see Missingness).

The Riegel curve is per-athlete, fit on `[0, warmup_days + train_days)` per ADR 0004. Train-period only; no leakage from test or validation windows. Per ADR 0004, warm-up days are included in the Riegel fit (warm-up exclusion is an A8 concern for the state estimator, not for static curve fitting).

The extracted best effort yields:

- `best_effort_riegel_speed_score` — observed best-effort speed divided by the athlete's Riegel-predicted speed at the best effort's distance. Dimensionless (~1.0 at curve, >1 above, <1 below).
- `best_effort_grade` — signed average grade over the best effort, `(elev_gain − elev_lost) / distance`. Dimensionless. Conditions the speed score: a given Riegel score at +4% grade is not the same capability readout as at flat.
- `best_effort_mean_HR` — mean HR over the best effort.
- `best_effort_hr_drift` — mean HR in the second half of the best effort minus mean HR in the first half (signed). Half-split by time (second half = samples with timestamp past the midpoint of the effort's duration).
- `best_effort_speed_cadence_ratio` — mean speed divided by mean cadence over the best effort. Stride-length proxy.

### `π_stim(X_t)` — transition projection (consumed by `f` at M5)

Whole-session aggregates, raw (no P-residualization — the stimulus *is* the realized load):

- `hr_load` — Edwards-TRIMP HR load over the session. Provided per session by the upstream data pipeline.
- `step_load` — speed-weighted step product over the session. Provided per session by the upstream data pipeline.
- `total_elevation_gain` — cumulative positive elevation change over the session.
- `total_elevation_lost` — cumulative negative elevation change over the session.
- `heat_exposure` — session-integrated heat exposure, `∫ (wet_bulb − 18°C)_+ dt`. Provided per session by the upstream data pipeline.

## Family

- **Likelihood:** Gaussian, `p(X̃ | Z, P, E) = N(μ(Z, P, E), Σ)`, with full covariance `Σ` across all X components. `X̃` is the pre-transformed X (below).
- **Mean function:** linear in `(Z, P, E)`, no interactions: `μ = A·Z + B·P + C·E + d`.
- **d_Z:** 4.
- **Pre-transforms (`X → X̃`):**
  - `heat_exposure → log(1 + heat_exposure)` — handles zero mass from cool-weather sessions.
  - `total_elevation_gain → log(1 + total_elevation_gain)` — handles zero mass on flat sessions and right skew.
  - `total_elevation_lost → log(1 + total_elevation_lost)` — same.
  - `hr_load → hr_load / 7200` — normalized to a 1hr zone-2 reference session (60 min × weight 2 × 60 sec/min).
  - `step_load → step_load / 16200` — normalized to a 1hr reference session at 3 m/s with 90 steps/min per foot (one-foot cadence convention).
  - `best_effort_riegel_speed_score → log(score)` — centers near 0 and symmetrizes tails around curve-matching.
  - Other components (`best_effort_grade`, `best_effort_mean_HR`, `best_effort_hr_drift`, `best_effort_speed_cadence_ratio`) enter raw.
- **Z-gauge:** deferred to the estimator reference impl (M5/M6).

## Missingness

No missingness handling in `π_obs`. A session qualifies as an observation-contributing workout day iff it contains at least one valid effort (see `π_obs` above). Sessions without a valid effort are not observation-contributing; they do not produce a `π_obs` readout for that day.

`π_stim` components (`hr_load`, `step_load`, `total_elevation_gain`, `total_elevation_lost`, `heat_exposure`) are provided per session by the upstream data pipeline; QA filtering upstream is assumed. Days with no session are rest days and route through `g`.

## Coherence with other impls

- `π_stim` is the input that `f` (the workout transition, M5) consumes from this observation impl. The M5 reference impl's spec must be coherent with `{hr_load, step_load, total_elevation_gain, total_elevation_lost, heat_exposure}` as `f`'s stimulus input.

## What would invalidate this

- Riegel breaks for the cohort (ultras, sprints, trail-dominated): the per-athlete Riegel fit produces unreliable predicted speeds, so `best_effort_riegel_speed_score` loses interpretability as a capability readout.
- Best-effort extraction is unstable: on sessions with near-tied candidates (efforts within a narrow margin of the maximum `observed_speed / riegel_speed(d)`), segment selection swings across sessions and adds extraction noise to `π_obs`.
- A large fraction of workout days contain no valid effort under the `[d_min, d_max]` gate: the qualify-or-drop rule reduces the sub-sample of observation-contributing sessions below what the estimator needs.
- Cadence signal quality insufficient for `best_effort_speed_cadence_ratio` to carry state information (e.g., wrist-optical or GPS-derived cadence too noisy over the train window).
- Elevation double-counting between `hr_load` and `π_stim` elevation components: climbing cost is absorbed into `hr_load`, and `total_elevation_gain/lost` coefficients in `f` collapse to near zero across the cohort.
- `best_effort_grade` near-zero on flat-dominant cohorts (track-heavy, treadmill-heavy): grade's coefficient identified only from the hilly-session sub-sample.
- HR drift confounded by within-effort non-stationarity: the simple-form drift (2nd-half − 1st-half mean HR) assumes the effort was approximately steady-state. Progressive efforts (intentional build), negative-split efforts (faster 2nd half), and rolling-terrain efforts (pace varies with grade) all produce 2nd-half HR differences for reasons other than cardiac drift.
- Riegel curve drifts across the year: the per-athlete curve is fit once on the train window and used as the denominator for `best_effort_riegel_speed_score` throughout test and validation. If the athlete's Riegel exponent or intercept drifts materially (fitness change, periodization regime, injury-recovery), test-window speed scores are biased against a stale curve. Low concern for this reference impl given the 60-day test window and train-window cohort stability assumptions (ADR 0004).
