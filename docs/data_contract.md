# Data Contract

Expected input schemas for `run_modeling`. Only the columns listed here are
consumed by the package; upstream dataframes may carry additional columns
that are ignored.

This document is schema only. Conventions are in `docs/conventions.md`.
DAG-level definitions are in `docs/theoretical_framework.md`.

---

## Session-level table (one row per activity)

| Column | Type | Required | Notes |
|---|---|---|---|
| `subject_id` | str | yes | Stable across activities for a given athlete |
| `activity_num` | int | yes | Monotone per subject |
| `date` | datetime64[ns] | yes | Calendar day |
| `distance_m` | float | yes | Session distance in meters |
| `effort_start_s` | float | yes | Elapsed seconds from activity start to the effort window |
| `best_5k_hr` | float | conditional | Required for CCT reference scoring |
| `speed_mps` | float | yes | Mean session speed |
| `hr_bpm` | float | yes | Mean session HR |
| `hr_max` | float | yes | Athlete's max HR (per-subject scalar, repeated) |
| `elevation_m` | float | yes | Mean elevation |
| `elev_gain` | float | yes | Cumulative elevation gain |
| `wet_bulb_temp` | float | conditional | If absent, derived from `air_temp_c` + `humidity_pct` |
| `tod_sin`, `tod_cos` | float | yes | Time-of-day encoding |
| `acwr_low`, `acwr_high` | float | yes | ACWR strata |

---

## 10-second step table (one row per step within a session)

| Column | Type | Required | Notes |
|---|---|---|---|
| `subject_id` | str | yes | |
| `activity_num` | int | yes | |
| `t_elapsed_s` | float | yes | Seconds from start of activity |
| `speed_mps` | float | yes | |
| `hr_bpm` | float | yes | |
| `hr_frac` | float | yes | `hr_bpm / hr_max` |
| `cardiac_cost_logit` | float | yes | `logit(hr_frac)` |
| `net_grade` | float | yes | Signed grade |
| `log_t0` | float | yes | `log(t_elapsed_s)` |
| `wet_bulb_temp` | float | yes | |
| `elevation_m` | float | yes | |
| `chronic_wet_bulb` | float | yes | 28-day rolling |
| `chronic_elevation` | float | yes | 28-day rolling |
| `roll_L1_step`, `roll_L2_step`, `roll_L3_step` | float | yes | 1-7d / 8-28d / 29-91d load windows |
| `tod_sin`, `tod_cos` | float | yes | |
