# Conventions

Cross-cutting conventions for `statepace`. The DAG itself is in
`docs/theoretical_framework.md`; the data schema is in `docs/data_contract.md`.

## Time and aggregation

- **Time index** is daily (DAG §A6). Multi-workout days aggregate before entering `X_t`.
- **Within-session resampling**: 10-second steps for any code operating on within-session dynamics.

## Warm-up and bounds

- **Warm-up mask** (per athlete): no scored predictions within an initial per-athlete window. Length (`warmup_days`) is a harness-level hyperparameter (§A8); protects against diffuse-prior residue.
- **Rest-day bound**: maximum consecutive rest days for the rest-day transition `g` (§A5). Length (`max_consecutive_rest_days`) is a hyperparameter on `RestTransition`. State is undefined beyond.

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
