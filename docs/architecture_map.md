# Architecture Map

Module skeleton for `statepace`, mapped to the DAG in
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
| `channels.py` | Primitives §1 (`Z`, `P`, `X`, `E`) — the typing layer | Session-vs-athlete boundary. `AthleteMeta` not on any DAG edge. Used only by split/cohort machinery in `evaluation/harness.py`. |
| `observation.py` | Edges `Z_{t-1} -> X_t`, `P_t -> X_t`, `E_t -> X_t` — i.e. `p(X_t \| Z_{t-1}, P_t, E_t)` (§3 observation model) | Carries forward (emission) + inverse (solve for the unobserved `X`-component given the rest). |
| `transitions.py` | Edges `Z_{t-1} -> Z_t`, `X_t -> Z_t` — `f` (workout) and `g` (rest, bounded per §A5 via `max_consecutive_rest_days`) | Two regimes, distinct functional forms. |
| `filter.py` | Inference target §6 — `p(Z_t \| P_{1:t}, X_{1:t}, E_{1:t})` (filter) and `p(Z_t \| P_{1:T}, X_{1:T}, E_{1:T})` (smoother) | One interface; estimator family (EWM / Kalman / GP / ML) is an implementation choice. |
| `forward.py` | τ-step predictive `p(Z_{t+τ-1} \| history)` using `f`, `g` from `transitions.py` (§7 first factor) | Closed-form only for degenerate `f`; general case is a sampler/propagator. |
| `predict.py` | Composition of §7 — filter -> forward -> observation inverse at queried `(P_{t+τ}, E_{t+τ})` | Pure composition; owns no DAG edge of its own. |
| `evaluation/harness.py` | No DAG node — runs fit/predict over fixtures, enforces warm-up mask (§A8, conventions §warm-up) and rest-day bound (§A5) | Eval-side only. |
| `evaluation/deconfounding.py` | Scoring convention only — projection to reference `(P*, E*)` (conventions §deconfounding). NOT an edge; NOT in the model. | Operates on observation-model outputs, never on `Z`. |
| `evaluation/metrics.py` | No DAG node — scoring functions | Eval-side only. |

Selection model `p(P_t \| Z_{t-1})` (§3, §5 mediator note) is **not**
given a module this round — see §5 below.

---

## 2. Skeleton

```
statepace/
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
from typing import Literal, Protocol
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


@dataclass(frozen=True)
class AthleteMeta:
    """Per-athlete invariant metadata. Not on any DAG edge.

    `Channels` is session-level (one row per day per athlete). `AthleteMeta`
    is athlete-level: one record per `subject_id`, stable across time. Used
    by split/cohort machinery (evaluation/harness.py) for stratified cohort
    assignment; never by observation, transitions, filter, forward, or
    predict.
    """
    subject_id: str
    sex: Literal["F", "M"]


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
`ChannelAssignment` Protocol; per-athlete invariant metadata
(`AthleteMeta`) as a sibling container to `Channels`.

**Out-of-scope**: any modeling; any use of `Z`; any derivation beyond
typing and daily alignment; estimator-specific feature engineering;
computed summaries over `Channels` (e.g., training volume) — those are
computed by the caller that consumes `AthleteMeta` (e.g.,
`evaluation/harness.py` at cohort-assignment time), not cached on
`AthleteMeta`.

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
from statepace.channels import P, X, E, Z, Array

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
up to `max_consecutive_rest_days` consecutive rest days (§A5, conventions §bounds).

```python
from __future__ import annotations
from typing import Protocol
from statepace.channels import X, Z, Array


class WorkoutTransition(Protocol):
    """p(Z_t | Z_{t-1}, X_t). Edge Z_{t-1} -> Z_t with X_t -> Z_t."""
    d_Z: int
    def step(self, Z_prev: Z, X_t: X) -> Z: ...
    def log_prob(self, Z_prev: Z, X_t: X, Z_t: Z) -> Array: ...


class RestTransition(Protocol):
    """p(Z_t | Z_{t-1}). Edge Z_{t-1} -> Z_t on rest days.

    Contract: valid only for consecutive-rest-day counts in
    [1, max_consecutive_rest_days]. Beyond the bound, callers must
    treat Z as undefined (A5).
    """
    d_Z: int
    max_consecutive_rest_days: int
    def step(self, Z_prev: Z, n_rest_days: int) -> Z: ...
    def log_prob(self, Z_prev: Z, n_rest_days: int, Z_t: Z) -> Array: ...
```

**In-scope**: `f`, `g`, their parameters, their validity bounds; noise
distribution on the transition.

**Out-of-scope**: observation model; filtering/smoothing; re-entry
policy after past-bound gaps (caller's responsibility, surfaced via the
`max_consecutive_rest_days` contract).

**Depends on**: `channels`, `numpy`.

---

### 3.4 `filter.py`

Inference over `Z`. Single interface across estimator families (EWM,
Kalman, GP, ML). Estimator choice is deferred to later rounds.

M6 widening (ADR 0006): `StateEstimator.infer` now returns `ZPosterior`
(sealed ABC) instead of `Z`. M6 returns `GaussianZPosterior`
(mean + marginal cov from Kalman pass). Successor families define their
own concrete subclasses without re-widening.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Literal, Mapping
from statepace.channels import Channels, Array
from statepace.observation import ObservationModel
from statepace.transitions import WorkoutTransition, RestTransition

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


class ZPosterior(ABC):
    """Sealed ABC for filtered / smoothed Z posterior (M6 widening, ADR 0006).

    Successor estimator families define concrete subclasses; M7+ forward.py
    and predict.py dispatch on the ABC interface.

    Attributes:
        dates: datetime64[ns] array, shape (T,).
        d_Z: latent dimensionality.
    """
    dates: Array   # datetime64[ns], shape (T,)
    d_Z: int

    @abstractmethod
    def mean(self) -> Array: ...        # (T, d_Z)

    @abstractmethod
    def sample(self, n: int, rng) -> Array: ...    # (n, T, d_Z)

    @abstractmethod
    def marginal_log_pdf(self, z: Array) -> Array: ...    # (T,)


@dataclass(frozen=True)
class GaussianZPosterior(ZPosterior):
    """Gaussian Kalman-filter posterior (M6 concrete subclass).

    Attributes:
        _mean: shape (T, d_Z).
        cov: shape (T, d_Z, d_Z).
        dates: datetime64[ns], shape (T,).
    """
    _mean: Array          # (T, d_Z)
    cov: Array            # (T, d_Z, d_Z)
    dates: Array          # datetime64[ns], shape (T,)


class StateEstimator(Protocol):
    """p(Z_t | history) under shared parameters and a given observation/transition family.

    `fit` learns shared parameters across a cohort of athletes (ADR 0001) and
    returns a *parametric* fitted estimator — it does not retain per-athlete data.
    `infer` computes the Z trajectory for any athlete (seen or unseen at fit time)
    by re-running inference against their Channels under the frozen parameters.
    This makes training-cohort and validation-cohort inference structurally identical
    (ADR 0002). Concrete implementations decide whether they use observation.log_prob,
    transition.log_prob, both, or approximations thereof.

    `d_Z` must equal the observation model's and transitions' `d_Z` at fit time;
    mismatch is a fit-time error.
    """

    d_Z: int

    def fit(
        self,
        cohort: Mapping[str, Channels],
        observation: ObservationModel,
        workout_transition: WorkoutTransition,
        rest_transition: RestTransition,
        prior: Prior | Mapping[str, Prior],
    ) -> "StateEstimator": ...

    def infer(
        self,
        channels: Channels,
        mode: InferMode = "filter",
        prior: Prior | None = None,
    ) -> ZPosterior: ...    # widened from Z at M6 (ADR 0006)
```

**In-scope**: filtering `p(Z_t | P_{1:t}, X_{1:t}, E_{1:t})`; smoothing
`p(Z_t | P_{1:T}, X_{1:T}, E_{1:T})`; accepting an explicit `Prior` on
`Z_0` (A8); warm-up enforcement (conventions §warm-up) via the
harness — the estimator itself emits estimates for all `t` and leaves
warm-up masking to callers. `ZPosterior` ABC + `GaussianZPosterior` (M6).

**Out-of-scope**: `p(X|Z,P,E)` (observation); `f`/`g` (transitions);
τ-step forward prediction (`forward.py`); scoring; the framework-level
decision that the prior is diffuse (that's A8, not a filter choice).

**Depends on**: `channels`, `observation`, `transitions`, `numpy`, `torch` (M6 reference impl only).

---

### 3.5 `forward.py`

τ-step state-forward: propagate `p(Z_t | history)` to
`p(Z_{t+τ-1} | history)` using `f` and `g`. Per §7, this is the first
factor in the prediction decomposition.

M6 widening (ADR 0006): `forward_state` now accepts and returns `ZPosterior`
instead of `Z`. Body remains `...` (M7 implements).

```python
from __future__ import annotations
from statepace.filter import ZPosterior
from statepace.transitions import WorkoutTransition, RestTransition


def forward_state(
    Z_t: ZPosterior,
    schedule: "ForwardSchedule",
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> ZPosterior: ...


from dataclasses import dataclass
from statepace.channels import X as X_channel, Array

@dataclass(frozen=True)
class ForwardSchedule:
    """τ-day schedule of workout/rest days ahead of t.

    For workout days, X_future provides the (hypothetical or planned) X
    driving f. For rest days, X_future rows are masked; consecutive rest
    counts are derived internally and must respect A5's rest bound (`max_consecutive_rest_days`).
    """
    is_rest: Array          # bool, shape (tau,)
    X_future: X_channel     # shape (tau, d_X); NaN on rest days
```

**In-scope**: τ-step propagation; rest-day counting and A5 bound check;
noise accumulation.

**Out-of-scope**: defining `f`/`g` (that's `transitions.py`); choosing
`X_future` (caller's question); observation inversion (`predict.py`).

**Depends on**: `channels`, `filter`, `transitions`.

---

### 3.6 `predict.py`

Composition of the prediction integral (§7):

```
p(X_{t+τ} | history) = ∫ p(X_{t+τ} | Z_{t+τ-1}, P_{t+τ}, E_{t+τ})
                       · p(Z_{t+τ-1} | history) dZ_{t+τ-1}
```

M6 scope: `predict_session` signature is stub-only. Body remains `...`
(M7 implements). M7 will wire `ZPosterior` from the filter as an argument —
that parameter design lands in M7.

```python
from __future__ import annotations
from statepace.channels import Channels, P, E, X, Array
from statepace.filter import StateEstimator, ZPosterior
from statepace.forward import ForwardSchedule
from statepace.observation import ObservationModel, ConditioningSpec
from statepace.transitions import WorkoutTransition, RestTransition


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
from typing import Iterable, Literal, Mapping, Sequence
from statepace.channels import AthleteMeta, Channels, Z, X, Array
from statepace.filter import StateEstimator, Prior
from statepace.observation import ObservationModel
from statepace.transitions import WorkoutTransition, RestTransition

Cohort = Literal["train", "test", "validation"]


@dataclass(frozen=True)
class EvalSplit:
    """Per-subject fit/score split with cohort label (ADR 0001/0002). Respects warm-up mask (conventions)."""
    subject_id: str
    cohort: Cohort
    fit_idx: Array    # int, indices into this athlete's Channels.dates
    score_idx: Array  # int, indices into this athlete's Channels.dates


# Stratified random cohort assignment per ADR 0003.
# Volume is summed over the train window [warmup_days, warmup_days+train_days);
# strata are (sex, volume bucket via np.digitize on volume_bucket_edges);
# validation_fraction is a per-stratum lower bound (ceil rounding).
def assign_cohorts(
    athletes: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    validation_fraction: float,
    seed: int,
    volume_bucket_edges: Sequence[float],
    warmup_days: int,
    train_days: int,
    volume_component: str,
) -> Mapping[str, Literal["train", "validation"]]: ...


# meta is keyed by subject_id matching cohort; cohort_assignment is the output of assign_cohorts.
# train_days and test_days are required to compute fit_idx / score_idx ranges.
def make_splits(
    cohort: Mapping[str, Channels],
    meta: Mapping[str, AthleteMeta],
    *,
    warmup_days: int,
    train_days: int,
    test_days: int,
    cohort_assignment: Mapping[str, Literal["train", "validation"]],
) -> Iterable[EvalSplit]: ...

def run_evaluation(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    estimator: StateEstimator,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
    *,
    prior: Prior | Mapping[str, Prior],
) -> "EvalResult": ...


@dataclass(frozen=True)
class EvalResult:
    Z_hat: Mapping[str, Z]                  # keyed by subject_id; estimator output on score_idx
    X_pred: Mapping[str, X]                 # keyed by subject_id; one-step observation predictions on score_idx
    cohort: Mapping[str, Cohort]            # subject_id -> cohort label
    rest_bound_violations: Mapping[str, Array]  # keyed by subject_id; bool, shape (T,), flags A5 overruns


# One concrete wiring of the Protocols; sweep-runner varies these across bundles
# while cohort and splits stay fixed. `label` keys results in SweepResult.
@dataclass(frozen=True)
class FormsBundle:
    label: str
    observation: ObservationModel
    workout_transition: WorkoutTransition
    rest_transition: RestTransition
    estimator: StateEstimator
    prior: Prior | Mapping[str, Prior]


# Invokes run_evaluation once per bundle; duplicate labels are a caller error.
# Body lands at M4 alongside the first reference ObservationModel.
def run_sweep(
    cohort: Mapping[str, Channels],
    splits: Iterable[EvalSplit],
    bundles: Sequence[FormsBundle],
) -> "SweepResult": ...


@dataclass(frozen=True)
class SweepResult:
    results: Mapping[str, EvalResult]    # keyed by bundle label
    bundles: Mapping[str, FormsBundle]   # keyed by bundle label; same keys as results
```

**In-scope**: fixture wiring (synthetic + real); stratified cohort
assignment (`assign_cohorts`, ADR 0003); warm-up enforcement;
rest-day-bound flagging; invoking estimator + observation to produce
arrays metrics operate on; sweeping across Protocol wirings
(`FormsBundle`, `run_sweep`, `SweepResult`) so the framework does not
privilege a single functional form.

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
from statepace.channels import P, E, Z, Array
from statepace.observation import ObservationModel, ConditioningSpec


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
from statepace.channels import X, Array
from statepace.evaluation.harness import EvalResult


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
   `statepace/`; `statepace/` must never import from
   `evaluation/`. Deconfounding, metrics, splits, harness are strictly
   downstream.
3. **One observation interface.** `ObservationModel` is a single
   Protocol; `Z` dimensionality varies *inside* implementations, never
   via branching in callers. `filter.py`, `forward.py`, `predict.py`
   treat `Z` opaquely via the Protocol.
4. **Deconfounding is scoring, not modeling.** `project_to_reference`
   lives in `evaluation/deconfounding.py`. No module in `statepace/`
   (observation, filter, transitions, forward, predict) may import it
   or depend on reference templates.
5. **Transitions own the rest bound.** `RestTransition.max_consecutive_rest_days`
   is the single source of truth for A5. Callers (`forward.py`,
   `filter.py`, eval harness) read it; they do not hardcode a literal.
6. **Warm-up is enforced at the edge.** Estimators emit `Z` over the
   full history; warm-up masking (conventions §warm-up) is applied by
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
12. **Estimator pooling is invisible to observation and transitions.**
    `StateEstimator.fit` takes a `Mapping[str, Channels]` (ADR 0001/0002),
    but `ObservationModel`, `WorkoutTransition`, and `RestTransition`
    continue to see per-athlete arrays only — estimators stack or loop
    internally. Pooling belongs to the estimator, not to the model
    components it composes.
13. **Session vs athlete-level boundary.** `Channels` is per-session
    (daily). `AthleteMeta` is per-athlete-invariant. No field may migrate
    across this boundary without a data-contract change. Per-athlete
    scalars that happen to appear row-replicated in the session table
    (e.g., `hr_max`) remain session-level in `Channels`; they are not
    duplicated onto `AthleteMeta`.

---

## 5. DAG primitives without a clean module home

Flagged, not fixed.

1. **Selection model `p(P_t | Z_{t-1})`** (§3 factorization, §5 mediator
   note). Required for causal statements but not for filtering or
   prediction given observed `P_{t+τ}`. No module in the skeleton owns
   it. If/when needed, it would plausibly sit as `selection.py`
   alongside `observation.py` with a symmetric Protocol
   (`forward: Z -> P`, `log_prob: (Z, P) -> array`). Deferred.
2. **Re-entry policy after past-bound rest gaps** (§A5). The contract
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
                          <-  forward  <-  predict   (forward also imports filter for ZPosterior)
                filter     <-  forward               (M6 widening: ZPosterior ABC)
         <-  observation  <-  forward (via predict)

evaluation/harness         <-  filter, observation, transitions, channels
evaluation/deconfounding   <-  observation, channels
evaluation/metrics         <-  evaluation/harness, evaluation/deconfounding
```

No cycles. `predict.py` is a pure composition node. `evaluation/` is a
strict sink. Note: `forward.py` imports `ZPosterior` from `filter.py` (M6
widening); `filter.py` does not import from `forward.py` — no cycle.
