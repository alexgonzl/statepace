# Conventions

Cross-cutting conventions for `run_modeling`. The DAG itself is in
`docs/theoretical_framework.md`; the data schema is in `docs/data_contract.md`.

## Time and aggregation

- **Time index** is daily (DAG §A6). Multi-workout days aggregate before entering `X_t`.
- **Within-session resampling**: 10-second steps for any code operating on within-session dynamics.

## Warm-up and bounds

- **1-year warm-up** per athlete: no scored predictions before 1 year of data. Tighter than the DAG's 6-month diffuse-prior bound (§A8); the extra margin protects against early-data instability.
- **Rest-day bound**: 10 consecutive rest days maximum for the rest-day transition `g` (§A5). State is undefined beyond.

## Deconfounding

- **Reference template** for projection: sea level, flat, 5k distance, noon TOD, 12°C wet-bulb.
- **Riegel weighting** belongs in fitting the observation model only. Evaluation is unweighted unless a specific estimand requires otherwise (state the estimand explicitly when it does).

## Modeling rules

See `CLAUDE.md` for the full standing-rules list; the load-bearing items:
- ACWR is an observation confounder, not a state.
- Chronic load is never a standalone regressor.
- No Banister assumptions.
- Cardiac drift is modeled, not truncated.
- Best-effort extraction: one effort per `activity_num` by Riegel score.
