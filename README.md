# statepace

Per-athlete cardiorespiratory state estimation from longitudinal run data.

Organized around the DAG in `docs/theoretical_framework.md`:

- Observation model `p(X | Z, P, E)`
- State transitions `f` (workout), `g` (rest, bounded)
- Prediction = state forward + observation at reference conditions

See `docs/theoretical_framework.md` for the framework and assumption ledger.

## Setup

```bash
conda env create -f environment.yml
conda activate statepace
```

The env includes `pip install -e .` so the package is importable in editable mode.

Quick check:

```bash
python -c "import statepace; print(statepace.__all__)"
```

## Layout

The package is being built against the DAG. Module skeleton per
`docs/architecture_map.md`:

```
statepace/
├── channels.py        # Z, P, X, E definitions; raw frame → typed channels
├── observation.py     # p(X | Z, P, E) forward + inverse
├── transitions.py     # f (workout), g (rest, bounded)
├── filter.py          # state estimators; single interface over Z
├── forward.py         # Z_d → Z_{d+τ} via f, g
├── predict.py         # filter → forward → invert
└── evaluation/
    ├── harness.py
    ├── deconfounding.py   # reference-template projection (eval-side)
    └── metrics.py
```

Implementation of these modules is pending. The current package exposes
only shared constants (`_constants.py`); see `docs/theoretical_framework.md`
for the DAG and `docs/architecture_map.md` for the interface contracts.
