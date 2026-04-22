# Architecture Map

Module skeleton for `run_modeling`, mapped to the DAG in
`docs/theoretical_framework.md`. Interfaces are typed Python signatures
only; no bodies, no estimator choices. Read alongside
`docs/theoretical_framework.md`, `docs/data_contract.md`,
`docs/conventions.md`.

The three parallel research goals (daily `Z`, next-session prediction,
identifiability) all share this skeleton. Differences across goals live
*inside* module implementations (filter family, observation
parameterization), not in the module graph.

---

## 1. Module -> DAG primitive map

| Module | DAG primitive(s) | Notes |
|---|---|---|
| `channels.py` | Primitives §1 (`Z`, `P`, `X`, `E`) — the typing layer | Boundary between raw frames and everything else. No DAG edge lives here. |
| `observation.py` | Edges `Z_{t-1} -> X_t`, `P_t -> X_t`, `E_t -> X_t` — i.e. `p(X_t \| Z_{t-1}, P_t, E_t)` (§3 observation model) | Carries forward (emission) + inverse (solve for the unobserved `X`-component given the rest). |
| `transitions.py` | Edges `Z_{t-1} -> Z_t`, `X_t -> Z_t` — `f` (workout) and `g` (rest, bounded 10d per §A5) | Two regimes, distinct functional forms. |
| `filter.py` | Inference target §6 — `p(Z_t \| P_{1:t}, X_{1:t}, E_{1:t})` (filter) and `p(Z_t \| P_{1:T}, X_{1:T}, E_{1:T})` (smoother) | One interface; estimator family (EWM / Kalman / GP / ML) is an implementation choice. |
| `forward.py` | τ-step predictive `p(Z_{t+τ-1} \| history)` using `f`, `g` from `transitions.py` (§7 first factor) | Closed-form only for degenerate `f`; general case is a sampler/propagator. |
| `predict.py` | Composition of §7 — filter -> forward -> observation inverse at queried `(P_{t+τ}, E_{t+τ})` | Pure composition; owns no DAG edge of its own. |
| `evaluation/harness.py` | No DAG node — runs fit/predict over fixtures, enforces warm-up (§A8, conventions §warm-up) and rest-day bound (§A5) | Eval-side only. |
| `evaluation/deconfounding.py` | Scoring convention only — projection to reference `(P*, E*)` (conventions §deconfounding). NOT an edge; NOT in the model. | Operates on observation-model outputs, never on `Z`. |
| `evaluation/metrics.py` | No DAG node — scoring functions | Eval-side only. |

Selection model `p(P_t \| Z_{t-1})` (§3, §5 mediator note) is **not**
given a module this round — see §5 below.

---

## 2. Skeleton

```
run_modeling/
  channels.py
  observation.py
  transitions.py
  filter.py
  forward.py
  predict.py
  evaluation/
    __init__.py
    harness.py
    deconfounding.py
    metrics.py
```

---

## 3. Public interfaces

Python 3.10+ type hints. `Array = numpy.ndarray` throughout. `DataFrame
= pandas.DataFrame` appears only in `channels.py`.

### 3.1 `channels.py`

Typed containers for the DAG's observable/latent primitives and the
single place raw frames enter the package.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass(frozen=True)
class SessionFrame:
    """Typed view of the session-level table (docs/data_contract.md)."""
    subject_id: str
    dates: Array              # datetime64[ns], shape (T,)
    activity_num: Array       # int, shape (T,)
    raw: pd.DataFrame         # the original frame; read-only downstream


@dataclass(frozen=True)
class StepFrame:
    """Typed view of the 10-second step table (docs/data_contract.md)."""
    subject_id: str
    activity_num: Array       # int, shape (N,)
    t_elapsed_s: Array        # float, shape (N,)
    raw: pd.DataFrame


@dataclass(frozen=True)
class P:
    """Session shape (DAG §1). Observed."""
    values: Array             # shape (T, d_P)
    names: tuple[str, ...]


@dataclass(frozen=True)
class X:
    """Execution (DAG §1). Observed; absent on rest days."""
    values: Array             # shape (T, d_X); NaN rows on rest days
    names: tuple[str, ...]
    is_rest: Array            # bool, shape (T,)


@dataclass(frozen=True)
class E:
    """Exogenous environment (DAG §1, §A2)."""
    values: Array             # shape (T, d_E)
    names: tuple[str, ...]


@dataclass(frozen=True)
class Z:
    """Latent state (DAG §1, §A4). Dimension is per-family."""
    mean: Array               # shape (T, d_Z)
    cov: Array | None         # shape (T, d_Z, d_Z) or None if point estimate
    dates: Array              # datetime64[ns], shape (T,)


@dataclass(frozen=True)
class Channels:
    """Bundle of typed channels aligned on a common daily index."""
    subject_id: str
    dates: Array              # datetime64[ns], shape (T,), daily
    P: P
    X: X
    E: E


class ChannelAssignment(Protocol):
    """Per-family rule mapping data-contract columns to (P, X, E).

    The concrete mapping lives in docs/channel_assignment.md
    (Deliverable #2). This Protocol is the in-code contract.
    """
    def split(self, session: SessionFrame, step: StepFrame) -> Channels: ...


def load_session_frame(df: pd.DataFrame, subject_id: str) -> SessionFrame: ...
def load_step_frame(df: pd.DataFrame, subject_id: str) -> StepFrame: ...

def to_channels(
    session: SessionFrame,
    step: StepFrame,
    assignment: ChannelAssignment,
) -> Channels: ...
```

**In-scope**: schema validation against `docs/data_contract.md`; daily
aggregation (§A6); multi-workout fusion; rest-day marking; the
`ChannelAssignment` Protocol.

**Out-of-scope**: any modeling; any use of `Z`; any derivation beyond
typing and daily alignment; estimator-specific feature engineering.

**Depends on**: `numpy`, `pandas`, `docs/data_contract.md`.

---

### 3.2 `observation.py`

Observation model `p(X_t | Z_{t-1}, P_t, E_t)` — forward (emit) and
inverse (solve for a held-out `X`-component under an explicit
conditioning specification).

The inverse contract replaces the earlier string-keyed `target` with a
typed `ConditioningSpec` dataclass. §5 distinguishes the **direct**
effect of `Z_{t-1}` on `X_t` (with `P_t` held fixed — mediator
conditioned on) from the **total** effect (with `P_t` marginalized via
the selection model). A bare string cannot express which parents are
`fixed`, `projected-to-reference`, or `marginalized`; the spec does.

Choice: explicit `ConditioningSpec` dataclass (not per-family `X`
subclasses). Justification: the conditioning mode is a query-time
property (direct vs total, projected vs raw), orthogonal to the
per-family `X` dimensionality — binding it to `X`'s type would force
every family to enumerate a type per query.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal
from run_modeling.channels import P, X, E, Z, Array

Mode = Literal["fixed", "projected", "marginalized"]


@dataclass(frozen=True)
class ConditioningSpec:
    """Typed specification for inverse / counterfactual queries.

    For each parent of X_t in the observation model, declare how it enters
    the query. `held_out` names the X-component the inverse solves for.

    - `fixed`: value is supplied by the caller (via the corresponding
      argument to `inverse`); the conditional is evaluated at that value.
    - `projected`: value is replaced by the reference template
      (conventions §deconfounding) before evaluation. Eval-side only;
      `observation.py` accepts the mode and requires a template from the
      caller.
    - `marginalized`: value is integrated out under its generative
      distribution. For P this requires the selection model p(P | Z);
      for the held-out X-component it is the default (inverse *is* a
      marginalization). Marginalizing P is the "total effect" mode of §5
      and requires a selection model to be wired in (deferred, §5.1).

    The held-out X-component is implicitly marginalized; `x_components`
    governs the *observed* X-components used to condition the inverse.
    """
    held_out: str                 # name of the X-component being solved for
    p_mode: Mode                  # how P_t enters (typically "fixed")
    e_mode: Mode                  # how E_t enters (typically "fixed")
    z_mode: Mode                  # how Z_{t-1} enters (typically "fixed")
    x_components: Mode            # how the *other* X-components enter


class ObservationModel(Protocol):
    """Single interface across all Z-dimensionalities and parameterizations.

    Inverse takes X with one component masked (NaN, named by
    `spec.held_out`) and returns the MAP/mean value of that component
    under the conditioning declared by `spec`.
    """

    d_Z: int

    def fit(self, Z_prev: Z, P: P, X: X, E: E) -> "ObservationModel": ...

    def forward(self, Z_prev: Z, P: P, E: E) -> X: ...

    def inverse(
        self,
        Z_prev: Z,
        P: P,
        E: E,
        X_partial: X,
        spec: ConditioningSpec,
    ) -> Array: ...

    def log_prob(self, Z_prev: Z, P: P, E: E, X: X) -> Array: ...
```

**In-scope**: the `p(X|Z,P,E)` conditional; forward emission; inverse
solve under a typed `ConditioningSpec`; likelihood evaluation for use
by `filter.py`.

**Out-of-scope**: state transitions (`transitions.py`); any knowledge of
`f` or `g`; deconfounding / reference-template projection (that is
eval-side — `evaluation/deconfounding.py` supplies the template when
`spec` declares a `projected` parent); selection model `p(P|Z)`
(required if any callsite sets `p_mode="marginalized"`; deferred per
§5.1).

**Depends on**: `channels`, `numpy`.

---

### 3.3 `transitions.py`

State dynamics. `f` on workout days, `g` on rest days; `g` is valid for
up to 10 consecutive rest days (§A5, conventions §bounds).

```python
from __future__ import annotations
from typing import Protocol
from run_modeling.channels import X, Z, Array


class WorkoutTransition(Protocol):
    """p(Z_t | Z_{t-1}, X_t). Edge Z_{t-1} -> Z_t with X_t -> Z_t."""
    d_Z: int
    def step(self, Z_prev: Z, X_t: X) -> Z: ...
    def log_prob(self, Z_prev: Z, X_t: X, Z_t: Z) -> Array: ...


class RestTransition(Protocol):
    """p(Z_t | Z_{t-1}). Edge Z_{t-1} -> Z_t on rest days.

    Contract: valid only for consecutive-rest-day counts in [1, 10].
    Beyond the bound, callers must treat Z as undefined (A5).
    """
    d_Z: int
    max_consecutive_rest_days: int    # = 10 per A5
    def step(self, Z_prev: Z, n_rest_days: int) -> Z: ...
    def log_prob(self, Z_prev: Z, n_rest_days: int, Z_t: Z) -> Array: ...
```

**In-scope**: `f`, `g`, their parameters, their validity bounds; noise
distribution on the transition.

**Out-of-scope**: observation model; filtering/smoothing; re-entry
policy after >10-day gaps (caller's responsibility, surfaced via the
`max_consecutive_rest_days` contract).

**Depends on**: `channels`, `numpy`.

---

### 3.4 `filter.py`

Inference over `Z`. Single interface across estimator families (EWM,
Kalman, GP, ML). Estimator choice is deferred to later rounds.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal
from run_modeling.channels import Channels, Z, Array
from run_modeling.observation import ObservationModel
from run_modeling.transitions import WorkoutTransition, RestTransition

InferMode = Literal["filter", "smooth"]


@dataclass(frozen=True)
class Prior:
    """Explicit prior on Z_0 (§A8).

    Framework commits to diffuse; the concrete numerical realization is
    a per-family choice and must be auditable. `diffuse=True` asserts
    the A8 claim; `mean`/`cov` carry the numerical surrogate the
    estimator actually uses (e.g., large-variance Gaussian, or None for
    families with an improper-prior treatment). Estimators must respect
    `diffuse` semantics (warm-up masking in the harness remains the
    defence against residual prior bias).
    """
    d_Z: int
    diffuse: bool                 # must be True to satisfy A8
    mean: Array | None            # shape (d_Z,) or None
    cov: Array | None             # shape (d_Z, d_Z) or None


class StateEstimator(Protocol):
    """p(Z_t | history) under a given observation model and transitions.

    Concrete implementations decide whether they use observation.log_prob,
    transition.log_prob, both, or approximations thereof.
    """
    def fit(
        self,
        channels: Channels,
        observation: ObservationModel,
        workout_transition: WorkoutTransition,
        rest_transition: RestTransition,
        prior: Prior,
    ) -> "StateEstimator": ...

    def infer(self, mode: InferMode = "filter") -> Z: ...
```

**In-scope**: filtering `p(Z_t | P_{1:t}, X_{1:t}, E_{1:t})`; smoothing
`p(Z_t | P_{1:T}, X_{1:T}, E_{1:T})`; accepting an explicit `Prior` on
`Z_0` (A8); warm-up enforcement (1 year, conventions §warm-up) via the
harness — the estimator itself emits estimates for all `t` and leaves
warm-up masking to callers.

**Out-of-scope**: `p(X|Z,P,E)` (observation); `f`/`g` (transitions);
τ-step forward prediction (`forward.py`); scoring; the framework-level
decision that the prior is diffuse (that's A8, not a filter choice).

**Depends on**: `channels`, `observation`, `transitions`, `numpy`.

---

### 3.5 `forward.py`

τ-step state-forward: propagate `p(Z_t | history)` to
`p(Z_{t+τ-1} | history)` using `f` and `g`. Per §7, this is the first
factor in the prediction decomposition.

```python
from __future__ import annotations
from run_modeling.channels import Z
from run_modeling.transitions import WorkoutTransition, RestTransition


def forward_state(
    Z_t: Z,
    schedule: "ForwardSchedule",
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> Z: ...


from dataclasses import dataclass
from run_modeling.channels import X as X_channel, Array

@dataclass(frozen=True)
class ForwardSchedule:
    """τ-day schedule of workout/rest days ahead of t.

    For workout days, X_future provides the (hypothetical or planned) X
    driving f. For rest days, X_future rows are masked; consecutive rest
    counts are derived internally and must respect A5's 10-day bound.
    """
    is_rest: Array          # bool, shape (tau,)
    X_future: X_channel     # shape (tau, d_X); NaN on rest days
```

**In-scope**: τ-step propagation; rest-day counting and A5 bound check;
noise accumulation.

**Out-of-scope**: defining `f`/`g` (that's `transitions.py`); choosing
`X_future` (caller's question); observation inversion (`predict.py`).

**Depends on**: `channels`, `transitions`.

---

### 3.6 `predict.py`

Composition of the prediction integral (§7):

```
p(X_{t+τ} | history) = ∫ p(X_{t+τ} | Z_{t+τ-1}, P_{t+τ}, E_{t+τ})
                       · p(Z_{t+τ-1} | history) dZ_{t+τ-1}
```

```python
from __future__ import annotations
from run_modeling.channels import Channels, P, E, X, Array
from run_modeling.filter import StateEstimator
from run_modeling.forward import ForwardSchedule
from run_modeling.observation import ObservationModel, ConditioningSpec
from run_modeling.transitions import WorkoutTransition, RestTransition


def predict_session(
    history: Channels,
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
    schedule: ForwardSchedule,
    P_target: P,
    E_target: E,
    X_partial: X,
    spec: ConditioningSpec,
) -> Array: ...
```

**In-scope**: gluing filter output -> forward propagation -> observation
inverse at the queried conditions. No new modeling.

**Out-of-scope**: defining any DAG edge; deconfounding; scoring.

**Depends on**: `channels`, `filter`, `forward`, `observation`,
`transitions`.

---

### 3.7 `evaluation/harness.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from run_modeling.channels import Channels, Z, X, Array
from run_modeling.filter import StateEstimator
from run_modeling.observation import ObservationModel
from run_modeling.transitions import WorkoutTransition, RestTransition


@dataclass(frozen=True)
class EvalSplit:
    """Per-subject fit/score split. Respects 1-year warm-up (conventions)."""
    subject_id: str
    fit_idx: Array    # int, indices into Channels.dates
    score_idx: Array  # int, indices into Channels.dates


def make_splits(channels: Channels, warmup_days: int = 365) -> Iterable[EvalSplit]: ...

def run_evaluation(
    channels: Channels,
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> "EvalResult": ...


@dataclass(frozen=True)
class EvalResult:
    Z_hat: Z          # estimator output on score_idx
    X_pred: X         # one-step observation predictions on score_idx
    rest_bound_violations: Array  # bool, shape (T,), flags A5 overruns
```

**In-scope**: fixture wiring (synthetic + real); warm-up enforcement;
rest-day-bound flagging; invoking estimator + observation to produce
arrays metrics operate on.

**Out-of-scope**: any model logic; deconfounding projection; metric
definitions.

**Depends on**: `channels`, `filter`, `observation`, `transitions`.

---

### 3.8 `evaluation/deconfounding.py`

Reference-template projection (conventions §deconfounding). **Scoring
convention only.** Not an edge, not a module the model sees.

```python
from __future__ import annotations
from dataclasses import dataclass
from run_modeling.channels import P, E, Z, Array
from run_modeling.observation import ObservationModel, ConditioningSpec


@dataclass(frozen=True)
class ReferenceTemplate:
    """Projection target (conventions §deconfounding):
    sea level, flat, 5k distance, noon TOD, 12C wet-bulb."""
    P_ref: P          # shape (1, d_P)
    E_ref: E          # shape (1, d_E)


def project_to_reference(
    Z_hat: Z,
    observation: ObservationModel,
    template: ReferenceTemplate,
    spec: ConditioningSpec,
) -> Array: ...
```

The caller constructs `spec` with `p_mode="projected"` and
`e_mode="projected"` (and typically `z_mode="fixed"` on `Z_hat`);
`project_to_reference` supplies `template` as the projection values.

**In-scope**: producing a deconfounded scalar (or vector) from a fitted
observation model at the reference `(P*, E*)` for scoring.

**Out-of-scope**: any role in fitting `observation` or `filter`; any
role in `predict.py`. Called only from `evaluation/metrics.py` or
user-facing eval scripts.

**Depends on**: `channels`, `observation`.

---

### 3.9 `evaluation/metrics.py`

```python
from __future__ import annotations
from run_modeling.channels import X, Array
from run_modeling.evaluation.harness import EvalResult


def one_step_prediction_error(result: EvalResult, X_true: X) -> Array: ...
def tau_step_prediction_error(result: EvalResult, X_true: X, tau: int) -> Array: ...
def state_stability(result: EvalResult) -> Array: ...
```

**In-scope**: scoring functions on `EvalResult` plus ground-truth
channels.

**Out-of-scope**: model logic; template choice (lives in
`deconfounding.py`).

**Depends on**: `channels`, `evaluation.harness`, `evaluation.deconfounding`.

---

## 4. Boundary rules

1. **Raw-frame boundary.** `channels.py` is the only module where
   `pandas.DataFrame` appears as an input type. Every other module
   takes/returns the typed channel dataclasses (`P`, `X`, `E`, `Z`,
   `Channels`). No module below `channels.py` touches `df[...]`.
2. **Evaluation direction.** `evaluation/` may import from
   `run_modeling/`; `run_modeling/` must never import from
   `evaluation/`. Deconfounding, metrics, splits, harness are strictly
   downstream.
3. **One observation interface.** `ObservationModel` is a single
   Protocol; `Z` dimensionality varies *inside* implementations, never
   via branching in callers. `filter.py`, `forward.py`, `predict.py`
   treat `Z` opaquely via the Protocol.
4. **Deconfounding is scoring, not modeling.** `project_to_reference`
   lives in `evaluation/deconfounding.py`. No module in `run_modeling/`
   (observation, filter, transitions, forward, predict) may import it
   or depend on reference templates.
5. **Transitions own the 10-day bound.** `RestTransition.max_consecutive_rest_days`
   is the single source of truth for A5. Callers (`forward.py`,
   `filter.py`, eval harness) read it; they do not hardcode `10`.
6. **Warm-up is enforced at the edge.** Estimators emit `Z` over the
   full history; warm-up masking (1 year, conventions) is applied by
   `evaluation/harness.py` and not baked into `filter.py`.
7. **No backdoor references to old modules.** `cardiac_cost.py`,
   `riegel.py`, `state_estimation.py`, `gp_estimator.py`,
   `prediction_layer.py`, `effort_utils.py`, `deconfounding.py` are
   reference only; no imports from new modules into them.
8. **Channel schema is the data contract.** Any column change in
   `docs/data_contract.md` must be reflected in `channels.py` and the
   `ChannelAssignment` Protocol; no module downstream may introduce its
   own schema.
9. **Selection model, if added, lives outside these modules.** See §5.
10. **Conditioning is explicit.** `ConditioningSpec` is the only way to
    parameterize `ObservationModel.inverse`, `predict_session`, and
    `project_to_reference`. No string-keyed `target` arguments; no
    implicit "direct vs total" defaults. A spec with
    `p_mode="marginalized"` is invalid until a selection model is
    wired (§5.1); observation implementations must reject it.
11. **Prior is explicit.** `StateEstimator.fit` takes a `Prior`
    dataclass; A8's diffuse claim is carried by `prior.diffuse=True` and
    may be asserted by the harness before fit.

---

## 5. DAG primitives without a clean module home

Flagged, not fixed.

1. **Selection model `p(P_t | Z_{t-1})`** (§3 factorization, §5 mediator
   note). Required for causal statements but not for filtering or
   prediction given observed `P_{t+τ}`. No module in the skeleton owns
   it. If/when needed, it would plausibly sit as `selection.py`
   alongside `observation.py` with a symmetric Protocol
   (`forward: Z -> P`, `log_prob: (Z, P) -> array`). Deferred.
2. **Re-entry policy after >10-day rest gaps** (§A5). The contract
   states `Z` is undefined past the bound and re-entry uses the last
   valid `Z_t` with degraded confidence, but *how* confidence
   degrades is a modeling choice with no obvious home. Candidates:
   `transitions.py` (gap-aware `g`), `filter.py` (prior inflation on
   re-entry). Flag for identifiability-auditor / senior-scientist.
3. **`Z_0` prior** (§A8). Diffuse by assumption; the concrete prior
   (mean, covariance, parametric form) is needed by `filter.py` but is
   not itself a DAG edge. **Resolved** in this revision as an explicit
   `Prior` dataclass threaded through `StateEstimator.fit` (§3.4). The
   `diffuse=True` flag makes the A8 claim auditable at the callsite.
4. **Within-day aggregation (§A6)** for multi-workout days. Belongs
   logically in `channels.py` (only place the raw frame lives), but the
   *choice* of aggregation is a modeling decision, not a typing
   decision. Held in `channels.py` for now with the understanding that
   the aggregation function is a per-family parameter, not a hardcode.
5. **Dual role of `X_t`** (§4). `X_t` appears as both input to
   `observation.log_prob` (as the observation) and input to
   `WorkoutTransition.step` (as the intervention driving `Z_t`). The
   skeleton represents this correctly — same `X` object used in both
   places — but there is no module whose sole job is to assert the two
   uses are consistent. Identifiability concern, not an architecture
   concern. Flag.
6. **Health-state projection `h(Z_t)`** (§A9, §8). Not on the prediction
   path; belongs in a `projections.py` or as methods on user-facing
   analysis tooling. Deferred.

---

## 6. Dependency graph (intra-package)

```
channels  <-  observation  <-  filter  <-  predict
         <-  transitions  <-  filter
                          <-  forward  <-  predict
         <-  observation  <-  forward (via predict)

evaluation/harness         <-  filter, observation, transitions, channels
evaluation/deconfounding   <-  observation, channels
evaluation/metrics         <-  evaluation/harness, evaluation/deconfounding
```

No cycles. `predict.py` is a pure composition node. `evaluation/` is a
strict sink.
