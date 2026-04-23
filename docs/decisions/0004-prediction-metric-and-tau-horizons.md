# 0004: Prediction metric and τ-horizons

**Date:** 2026-04-22
**Status:** accepted

## Decision

### Residual streams

- **Raw-space** — prediction vs. observed `X` under the athlete's own `(P_t, E_t)`.
- **Reference-space** — both projected to the reference template (sea level, flat, 5k, noon, 12°C wet-bulb; `docs/conventions.md` §deconfounding) via `project_to_reference`, then compared.

### Metric target

A single designated scalar channel of `X` per evaluation — canonically speed; HR or another single channel via configuration. The concrete `X.names` entry lands at M9/M10 per late-naming. D10 pins only that the target is a single scalar, not a vector.

### Aggregated statistics (per subject, per cohort slot, per scope)

- RMSE
- MAE
- normRMSE (RMSE / mean of observed `X` on the scoring window)
- normMAE (MAE / mean of observed `X` on the scoring window)
- Spearman rank correlation (predicted vs. observed)

### Top-speed subset

Raw-space residuals restricted to scoring days where observed speed exceeds 90% of a per-athlete Riegel-scored speed at the session distance. The Riegel curve is fit per athlete on `[0, warmup_days + train_days)` — warm-up inclusive, test segment excluded. Warm-up exclusion is an A8 concern for the state estimator, not for static curve fitting. Consumes `RIEGEL_DISTANCES_M` from `_constants.py`; does not use `WORLD_RECORD_SPEEDS`.

Reported **pooled only**. τ-binning would yield ≈1.5 days per cell at 60 scoring days.

### τ-bins

τ = days between last history day and scoring day. Bins: `1–3, 4–7, 8–10, 11+`. Raw and reference streams only; top-speed excluded.

### Per-subject scalar tally

| Scope | Raw | Reference | Top-speed | Total |
|---|---|---|---|---|
| Pooled | 5 | 5 | 5 | 15 |
| τ-binned (×4) | 5×4 | 5×4 | — | 40 |
| | | | | **55** |

### Three-way cohort reporting

- **Train** — training-cohort athlete, train segment `[warmup_days, warmup_days + train_days)`. In-sample time, in-sample athletes.
- **Test** — training-cohort athlete, test segment. Out-of-sample time.
- **Val** — validation-cohort athlete, test segment. Out-of-sample time and athletes.

A subject contributes to either `{train, test}` or `{val}`; athlete-disjoint per ADR 0001/0002/0003.

## Context

D10 in `docs/plans/post-scaffold-sequencing.md` — needed before M8 wires `evaluation/metrics.py`. Architecture map §3.9 declares metric signatures and defers content here.

Headline claims: (1) predictive accuracy of the next session, (4) at reference conditions. Raw-space and reference-space cover these respectively. Both reported because divergence between them is diagnostic.

## Alternatives considered

- **Blended scalar across raw and reference.** Rejected — the divergence is the most informative diagnostic; blending hides it.
- **Per-day residual arrays.** Rejected — cross-bundle comparison at M8 needs scalars. Per-day `X_pred` remains on `EvalResult` for on-demand recovery.
- **Two-way cohort reporting.** Rejected — the train slot attributes generalization gaps to learning vs. cross-athlete transfer.
- **Top-speed τ-binned.** Rejected — sparsity.

## DAG implications

No DAG edge added, removed, or altered. Metrics consume `EvalResult`. Reference-space depends on `project_to_reference`, which is scoring convention (boundary rule 4). A1–A10 unchanged.

## What would invalidate this

- **`project_to_reference` produces artifacts** (extrapolation beyond observed range). Drop reference-space for affected subjects.
- **`project_to_reference` is non-injective** — distinct raw predictions collapsing to the same reference point would artificially tighten reference-space residuals. Surface projection-rank diagnostics at M8.
- **Riegel breaks for the cohort** (ultras, sprints, trail). Re-scope or remove top-speed.
- **Top-speed cells sparse even pooled** at real-cohort volumes.
- **τ-bin cells imbalanced** — dominated by one bin, breakdown conveys no information beyond pooled.
- **Train-slot residuals carry A8 prior residue.** Surface first-N-days-of-train residuals as a diagnostic at M10.

## Follow-ups

- **`make_splits` emits an in-sample split.** For each training-cohort athlete, an additional `EvalSplit` with `cohort="train"` and `score_idx` inside `[warmup_days, warmup_days + train_days)`. Lands at M4 with the `run_evaluation` body; existing tests updated in the same commit.
- **`project_to_reference` body required.** `evaluation/deconfounding.py` is signature-only. Body lands before or alongside M8.
- **Top-speed distance dependency.** Riegel lookup needs a named distance component of `P`; name deferred to M9/M10.
- **Weather / load breakdowns.** Secondary exploration; separate scoping round.
- **Plan doc sync.** `docs/plans/post-scaffold-sequencing.md` flips D10 to resolved, links this ADR, records the `make_splits` and `project_to_reference` follow-ups.
