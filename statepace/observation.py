"""Observation model p(X_t | Z_{t-1}, P_t, E_t): forward emission and typed inverse (architecture_map §3.2)."""
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
