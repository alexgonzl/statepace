# Channel Assignment

Per-column mapping of the `docs/data_contract.md` schemas onto the DAG
primitives of `docs/theoretical_framework.md` (§1 primitives; §A1 `P/X`
split; §A2 exogeneity of `E`; §A3 no-direct-edge structure). This is the
concrete realization of the `ChannelAssignment` Protocol in
`channels.py` (architecture_map §3.1) that the typed `ConditioningSpec`
(§3.2) must dispatch over.

Buckets:

- `Z` — latent state component (unobserved by construction; no raw column can live here)
- `P` — session shape / planned condition (DAG §1, §A1)
- `X` — execution / performance-bearing measurement (DAG §1, §A1)
- `E` — genuinely exogenous environment (DAG §1, §A2)
- `evaluation-only` — never consumed by model; scoring, alignment, or diagnostic
- `ambiguous: needs identifiability-auditor input` — flagged tension

---

## Session-level table

| Column | Bucket | Justification |
|---|---|---|
| `subject_id` | evaluation-only | Identity / indexing key; not a DAG node (framework §A7: per-athlete DAG, between-athlete variation lives in `Z`, not in an observable channel). |
| `activity_num` | evaluation-only | Monotone session counter; indexing only, not a parent of `X_t` under DAG §2. |
| `date` | evaluation-only | Daily index `t` (§1, §A6); aligns rows, not a DAG node. |
| `distance_m` | P | Session-shape channel chosen by the athlete before execution (§A1: "what the athlete set out to do"); mediator of `Z_{t-1} → X_t` per §5. |
| `effort_start_s` | P | Structural offset into the planned session (where the scored effort begins); part of the frame, not the execution (§A1). |
| `best_5k_hr` | evaluation-only | Per-subject reference scalar for CCT scoring; consumed by `evaluation/deconfounding.py` reference template (architecture §3.8), never enters the model. |
| `speed_mps` (session) | X | Mean realized speed — produced during execution, the canonical performance-bearing signal (§1 `X_t`; §4 duality). |
| `hr_bpm` (session) | X | Body-produced cardiovascular response during execution (§1 `X_t`). |
| `hr_max` | evaluation-only | Per-subject stable trait — §8 places stable individual traits *inside* `Z` (§A7); as a raw column it is used only to normalize HR (e.g., `hr_frac`), not as a DAG parent. |
| `elevation_m` (session) | P | Mean elevation of the chosen route; route is selected (§A2 scope note explicitly places route/terrain in `P`, not `E`). |
| `elev_gain` | P | Cumulative climb of the selected route; same rationale as `elevation_m` — route-structural, not executed (§A2 scope note). |
| `wet_bulb_temp` (session) | E | Ambient thermal load; §A2 exogeneity holds — weather is not selected for its interaction with `Z_{t-1}`. |
| `tod_sin`, `tod_cos` (session) | P | Start-time is a planned session-shape component (A2: planned condition, not ambient environment). The alternative `E` assignment is rejected on selection-on-observables grounds per `docs/identifiability_baseline.md` §4 concern 3 — admitting as `E` would require an empirical selection audit each run. |
| `acwr_low`, `acwr_high` | evaluation-only | Deterministic projection of past `X_{1:t-1}`; admitting as `P` or `E` would break A2 exogeneity and the §A3 no-direct-edge claim. Consumed only by `evaluation/deconfounding.py` as nuisance covariate (CLAUDE.md ACWR rule; §11). |

---

## 10-second step table

| Column | Bucket | Justification |
|---|---|---|
| `subject_id` | evaluation-only | Indexing key; same rationale as session-level. |
| `activity_num` | evaluation-only | Indexing key joining step rows to a session. |
| `t_elapsed_s` | X | Intra-session time coordinate of the execution trace; parameterizes the realized trajectory `X_t` (§A6 within-day structure collapses under daily index but is carried inside `X_t`). |
| `speed_mps` (step) | X | Instantaneous realized speed during execution (§1 `X_t`). |
| `hr_bpm` (step) | X | Instantaneous HR during execution (§1 `X_t`). |
| `hr_frac` | X | `hr_bpm / hr_max` — monotone reparameterization of an `X` channel; still execution-produced (§1). |
| `cardiac_cost_logit` | X | `logit(hr_frac)` — reparameterization of the execution HR signal used as the observation-model target (§3 observation model `p(X | Z, P, E)`). |
| `net_grade` | P | Instantaneous grade of the selected route; route/terrain is `P` per §A2 scope note, not `E`. |
| `log_t0` | X | `log(t_elapsed_s)` — reparameterization of the execution time coordinate; same bucket as `t_elapsed_s`. |
| `wet_bulb_temp` (step) | E | Ambient thermal load sampled within the session; exogenous per §A2. |
| `elevation_m` (step) | P | Instantaneous elevation of the selected route; §A2 places route/terrain in `P`. |
| `chronic_wet_bulb` | evaluation-only | Deterministic 28-day projection of past `E` shaped by past `P` (via exposure selection); admitting as `E` would break A2 exogeneity and §A3. Consumed only by `evaluation/deconfounding.py` as nuisance covariate (§8 absorbs chronic adaptation into `Z`). |
| `chronic_elevation` | evaluation-only | Deterministic 28-day projection of past `P`; no native bucket (chronic-`P` summary). Consumed only by `evaluation/deconfounding.py` as nuisance covariate (§8). |
| `roll_L1_step`, `roll_L2_step`, `roll_L3_step` | evaluation-only | 1–7d / 8–28d / 29–91d rolling load windows — deterministic projections of past `X_{1:t-1}`; admitting as `P` or `E` would break A2 and §A3 (§11; CLAUDE.md ACWR rule). Consumed only by `evaluation/deconfounding.py`. |
| `tod_sin`, `tod_cos` (step) | P | Start-time is a planned session-shape component (A2: planned condition, not ambient environment); same rationale as session-level ToD. Alternative `E` assignment rejected on selection-on-observables grounds per `docs/identifiability_baseline.md` §4 concern 3. |

---

## Bucket counts

- `Z`: 0 (latent by definition; no raw column lives here)
- `P`: 8 — `distance_m`, `effort_start_s`, `elevation_m` (session), `elev_gain`, `tod_sin`/`tod_cos` (session), `net_grade`, `elevation_m` (step), `tod_sin`/`tod_cos` (step)
- `X`: 8 — `speed_mps` (session), `hr_bpm` (session), `t_elapsed_s`, `speed_mps` (step), `hr_bpm` (step), `hr_frac`, `cardiac_cost_logit`, `log_t0`
- `E`: 2 — `wet_bulb_temp` (session), `wet_bulb_temp` (step)
- `evaluation-only`: 14 — `subject_id` (×2), `activity_num` (×2), `date`, `best_5k_hr`, `hr_max`, `acwr_low`, `acwr_high`, `chronic_wet_bulb`, `chronic_elevation`, `roll_L1_step`, `roll_L2_step`, `roll_L3_step`
- `ambiguous`: 0

(Paired ToD and paired step-rolling-load columns counted once per conceptual channel.)
