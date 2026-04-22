"""State inference protocol and Prior container for p(Z_t | history) (architecture_map §3.4)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal
from statepace.channels import Channels, Z, Array
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
