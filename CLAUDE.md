# CLAUDE.md — statepace

Research codebase. Scientific correctness > engineering polish.

---

## Read First

| Document | When |
|---|---|
| `docs/theoretical_framework.md` | Before any modeling decision — the DAG is the guiding principle |
| `docs/data_contract.md` | Before touching I/O — expected input schemas |
| `docs/conventions.md` | For cross-cutting conventions (warm-up, bounds, deconfounding template) |
| `README.md` | For layout + setup |

---

## What This Project Does

Per-athlete cardiorespiratory state estimation from longitudinal run data:

- **Observation model** `p(X | Z, E)` — `cardiac_cost.py`, `riegel.py`, `effort_utils.py`, `deconfounding.py`
- **State transitions** `f` (workout), `g` (rest, bounded 10d) — `state_estimation.py`, `gp_estimator.py`
- **Prediction** = state forward + observation at reference conditions — `prediction_layer.py`

The cardiac cost work is the most recent anchor; see the LMM cascade and figures in the old `consolidated_cardiac_cost_report.md`.

---

## Standing Rules

- No unsolicited files, no scope creep, no renormalization unless asked.
- Preserve public APIs (what `__init__.py` exports) unless explicitly asked to change.
- Test with real fixture data; no mocking internals.
- No IDE config files.
- Commit discipline: never commit without a runtime test or explicit acknowledgement of risk.
- No dead parameters — remove unused/obsolete when refactoring.
- Report numbers and deltas; don't label pass/fail/success.

---

## User Preferences

- Python is primary.
- To-the-point, succinct answers. Minimize explanation unless asked.
- Always show math when appropriate.
- Ask before opening unnecessary files or running tools that fill context.
- One question at a time with a recommendation when scoping.

---

## Key Modeling Decisions (preserved from predecessor)

- `RiegelHRModel(predictor='dist', family='qr', equalize_distances=True)` is the preferred L1 model.
- `q=0.9` for fitness ceiling; quantile family used diagnostically only.
- Best-effort extraction: one effort per `activity_num` by Riegel score (independence guarantee).
- `solver='highs'` required for sklearn 1.1.3 QuantileRegressor.
- EWM with `halflife='28D'`, `times=` for calendar-aware decay, `min_periods=7`.
- Projection uses `fit_data_` (best efforts), not all efforts.
- ACWR is observation confounder, not state — never include chronic load as a standalone regressor.
- No Banister assumptions — load proxies are not identified fitness/fatigue states.
- Cardiac drift is modeled, not truncated.

---

## Environment

```bash
conda env create -f environment.yml
conda activate statepace
python -c "import statepace; print(len(statepace.__all__), 'exports')"
```
