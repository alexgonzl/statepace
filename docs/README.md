# docs/ — organization schema

This folder has a fixed layout. Follow it for every new doc, script, or
figure.

```
docs/
├── README.md                          ← this file
├── theoretical_framework.md           ← prose: the DAG and assumption ledger
├── conventions.md                     ← prose: cross-cutting conventions
├── data_contract.md                   ← prose: expected input schemas
├── dag_buildup.md                     ← prose: step-by-step DAG narration
├── architecture_map.md                ← prose: planned module layout and contracts
├── channel_assignment.md              ← prose: P/X/E classification of raw columns
├── identifiability_baseline.md        ← prose: minimal-model identifiability audit
├── decisions/                         ← ADRs via `modeling-decision-record` skill (created on first use)
├── plans/                             ← sequencing plans; live-maintained per `plans/README.md`
├── reference_impls/                   ← one spec per reference implementation; governed by `reference_impls/README.md`
├── theoretical_framework_dag.{png,svg}   ← canonical framework figure (top-level)
├── scripts/                           ← render sources (executable .py)
│   ├── make_dag.py                       → writes ../theoretical_framework_dag.{png,svg}
│   └── make_dag_buildup.py               → writes ../figures/dag_buildup/step{N}.{png,svg}
└── figures/                           ← rendered figure outputs
    └── dag_buildup/
        └── step{1..6}.{png,svg}
```

## Rules

1. **Prose** (`*.md`) lives at `docs/` top level. No prose inside `scripts/`
   or `figures/`.
2. **Render scripts** (`make_*.py`) live in `docs/scripts/`. A render script
   must write its outputs to a path under `docs/` — never next to itself in
   `scripts/`. Resolve the output root as
   `Path(__file__).parent.parent` (= `docs/`).
3. **Rendered figures** live in `docs/figures/`, except the canonical
   `theoretical_framework_dag.{png,svg}` which stays at `docs/` top level as
   the framework figure referenced from `theoretical_framework.md`.
4. **Figure sequences** get their own subfolder under `docs/figures/`.
   Inside a sequence subfolder, drop the folder's name from filenames —
   e.g. `figures/dag_buildup/step1.png`, not
   `figures/dag_buildup/dag_buildup_step1.png`.
5. **Both PNG and SVG** for every rendered figure. PNG for slides and
   rendering, SVG for editability.
6. **One render script per figure family.** If the figure is a sequence,
   the script emits the whole sequence in one run.
7. **No stray outputs at `docs/` top level** other than the canonical DAG
   and the `*.md` files. Everything else routes through `figures/` or
   `scripts/`.

## Adding something new

| New thing | Where it goes |
|---|---|
| Prose doc | `docs/<name>.md` |
| Render script | `docs/scripts/make_<name>.py`, writing to `docs/figures/<name>/` (or `docs/figures/<name>.{png,svg}` for a single figure) |
| Single rendered figure | `docs/figures/<name>.{png,svg}` |
| Figure sequence | `docs/figures/<name>/step{N}.{png,svg}` |

If a new artifact doesn't fit the table, extend this README before adding
it.
