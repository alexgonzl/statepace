"""Observation model p(X_t | Z_{t-1}, P_t, E_t): forward emission and typed inverse (architecture_map §3.2)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal
import numpy as np
from scipy.stats import multivariate_normal
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


# ---------------------------------------------------------------------------
# Reference implementation: Riegel-score / HR+step load
# ---------------------------------------------------------------------------

# Component names as defined in the spec (riegel-score-hrstep.md §Channel composition).
_X_NAMES: tuple[str, ...] = (
    "best_effort_riegel_speed_score",
    "best_effort_grade",
    "best_effort_mean_HR",
    "best_effort_hr_drift",
    "best_effort_speed_cadence_ratio",
    "hr_load",
    "step_load",
    "total_elevation_gain",
    "total_elevation_lost",
    "heat_exposure",
)


def _transform(x_raw: Array, names: tuple[str, ...]) -> Array:
    """Apply per-component pre-transforms X → X̃ (spec §Family).

    Lookups are by name, not position, so the transform is robust to
    column reordering in the caller's X.names.

    Args:
        x_raw: shape (T, d_X), raw X values.
        names: component names aligned with columns of x_raw.

    Returns:
        x_tilde: shape (T, d_X), transformed values.
    """
    x_tilde = x_raw.copy().astype(float)
    n = {name: i for i, name in enumerate(names)}

    x_tilde[:, n["best_effort_riegel_speed_score"]] = np.log(
        x_raw[:, n["best_effort_riegel_speed_score"]]
    )
    x_tilde[:, n["hr_load"]] = x_raw[:, n["hr_load"]] / 7200.0
    x_tilde[:, n["step_load"]] = x_raw[:, n["step_load"]] / 16200.0
    x_tilde[:, n["total_elevation_gain"]] = np.log1p(x_raw[:, n["total_elevation_gain"]])
    x_tilde[:, n["total_elevation_lost"]] = np.log1p(x_raw[:, n["total_elevation_lost"]])
    x_tilde[:, n["heat_exposure"]] = np.log1p(x_raw[:, n["heat_exposure"]])
    # Passthrough: best_effort_grade, best_effort_mean_HR,
    #              best_effort_hr_drift, best_effort_speed_cadence_ratio.
    return x_tilde


def _inverse_transform_component(x_tilde_col: Array, name: str) -> Array:
    """Invert the pre-transform for a single named X component.

    Args:
        x_tilde_col: shape (T,), transformed values for one component.
        name: component name.

    Returns:
        shape (T,), raw-space values.
    """
    if name == "best_effort_riegel_speed_score":
        return np.exp(x_tilde_col)
    elif name in ("total_elevation_gain", "total_elevation_lost", "heat_exposure"):
        return np.expm1(x_tilde_col)
    elif name == "hr_load":
        return x_tilde_col * 7200.0
    elif name == "step_load":
        return x_tilde_col * 16200.0
    else:
        # Passthrough components: grade, mean_HR, hr_drift, speed_cadence_ratio.
        return x_tilde_col.copy()


class RiegelScoreHRStep:
    """Reference ObservationModel: Gaussian linear emission with Riegel-score + HR/step-load channels.

    Implements p(X̃ | Z_{t-1}, P_t, E_t) = N(A·Z + B·P + C·E + d, Σ) after
    per-component pre-transforms X → X̃ (spec §Family).

    Parameters are populated by `fit`; calling `forward`, `inverse`, or
    `log_prob` on an unfit instance raises RuntimeError.

    Design choice — `forward` returns is_rest=zeros:
        The Protocol's `forward` takes (Z_prev, P, E) without a rest mask.
        `forward` is a model prediction (conditional mean), not a data-bearing
        channel, so it is semantically all-workout. `is_rest` is therefore set
        to an all-False array of length T.

    Design choice — unfit raises RuntimeError:
        Preferred over None attributes because the error message identifies
        the method called and the fix needed, rather than surfacing as an
        AttributeError on a None dereference.
    """

    def __init__(self, d_Z: int) -> None:
        self.d_Z = d_Z
        self.A: Array | None = None   # (d_X, d_Z); e.g. (10, 4) at d_Z=4
        self.B: Array | None = None   # (d_X, d_P); e.g. (10, 6)
        self.C: Array | None = None   # (d_X, d_E); e.g. (10, 1)
        self.d: Array | None = None   # (d_X,); e.g. (10,)
        self.Σ: Array | None = None   # (d_X, d_X); e.g. (10, 10)
        self._x_names: tuple[str, ...] | None = None

    def _require_fit(self, method: str) -> None:
        """Raise RuntimeError if the model has not been fit.

        Args:
            method: name of the calling method, included in the error message.
        """
        if self.A is None:
            raise RuntimeError(
                f"RiegelScoreHRStep.{method} called before fit(). "
                "Call fit(Z_prev, P, X, E) first."
            )

    def fit(self, Z_prev: Z, P: P, X: X, E: E) -> "RiegelScoreHRStep":
        """Fit OLS parameters from cohort data.

        Transforms X → X̃, drops rest-day rows, builds design matrix
        Φ = [Z_prev.mean | P.values | E.values | 1] on non-rest rows, and
        solves multivariate OLS. Returns a new fitted instance; does not
        mutate self.

        Args:
            Z_prev: latent state, shape (T, d_Z).
            P: session shape, shape (T, d_P).
            X: execution channels, shape (T, d_X); rest rows are NaN.
            E: exogenous, shape (T, d_E).

        Returns:
            New RiegelScoreHRStep instance with A, B, C, d, Σ populated.
        """
        if Z_prev.mean.shape[1] != self.d_Z:
            raise ValueError(
                f"Z_prev.mean has d_Z={Z_prev.mean.shape[1]}, but "
                f"RiegelScoreHRStep was constructed with d_Z={self.d_Z}."
            )

        x_tilde_full = _transform(X.values, X.names)

        # Non-rest mask: rows where X is observed.
        active = ~X.is_rest                     # (T,) bool
        x_tilde = x_tilde_full[active]          # (N, d_X)
        z = Z_prev.mean[active]                 # (N, d_Z)
        p = P.values[active]                    # (N, d_P)
        e = E.values[active]                    # (N, d_E)
        N = x_tilde.shape[0]

        # Design matrix: [Z | P | E | 1], shape (N, d_Z + d_P + d_E + 1).
        ones = np.ones((N, 1), dtype=float)
        Phi = np.concatenate([z, p, e, ones], axis=1)   # (N, 12)

        # Multivariate OLS: Θ has shape (12, d_X).
        Theta, _, _, _ = np.linalg.lstsq(Phi, x_tilde, rcond=None)

        d_Z = Z_prev.mean.shape[1]
        d_P = P.values.shape[1]
        d_E = E.values.shape[1]

        # Split Θ into blocks aligned with Φ column partition.
        A_T = Theta[:d_Z, :]                      # (d_Z, d_X)
        B_T = Theta[d_Z: d_Z + d_P, :]            # (d_P, d_X)
        C_T = Theta[d_Z + d_P: d_Z + d_P + d_E, :]  # (d_E, d_X)
        d_vec = Theta[d_Z + d_P + d_E, :]         # (d_X,)

        A = A_T.T   # (d_X, d_Z)
        B = B_T.T   # (d_X, d_P)
        C = C_T.T   # (d_X, d_E)

        residuals = x_tilde - Phi @ Theta          # (N, d_X)
        n_params = Phi.shape[1]
        Sigma = (residuals.T @ residuals) / (N - n_params)  # (d_X, d_X)

        inst = RiegelScoreHRStep(d_Z=self.d_Z)
        inst.A = A
        inst.B = B
        inst.C = C
        inst.d = d_vec
        inst.Σ = Sigma
        inst._x_names = X.names
        return inst

    def _mean_tilde(self, Z_prev: Z, P: P, E: E) -> Array:
        """Compute conditional mean μ̃ in transformed space.

        Args:
            Z_prev: shape (T, d_Z).
            P: shape (T, d_P).
            E: shape (T, d_E).

        Returns:
            shape (T, d_X).
        """
        return (
            Z_prev.mean @ self.A.T
            + P.values @ self.B.T
            + E.values @ self.C.T
            + self.d
        )

    def forward(self, Z_prev: Z, P: P, E: E) -> X:
        """Return the conditional mean prediction in raw X space.

        Returns is_rest = zeros(T) because forward is a model prediction
        (no rest structure); see class docstring for rationale.

        Args:
            Z_prev: shape (T, d_Z).
            P: shape (T, d_P).
            E: shape (T, d_E).

        Returns:
            X dataclass with values shape (T, d_X) in raw space.
        """
        self._require_fit("forward")
        mu_tilde = self._mean_tilde(Z_prev, P, E)   # (T, d_X)
        T, d_X = mu_tilde.shape
        names = self._x_names

        raw = np.empty_like(mu_tilde)
        for j, name in enumerate(names):
            raw[:, j] = _inverse_transform_component(mu_tilde[:, j], name)

        return X(
            values=raw,
            names=names,
            is_rest=np.zeros(T, dtype=bool),
        )

    def inverse(
        self,
        Z_prev: Z,
        P: P,
        E: E,
        X_partial: X,
        spec: ConditioningSpec,
    ) -> Array:
        """Solve for the held-out X component via conditional Gaussian formula.

        Only supports spec with all modes == "fixed". Raises NotImplementedError
        otherwise.

        Given observed X̃_o and the held-out index h, computes:
            μ̃_{h|o} = μ̃_h + Σ_{h,o} @ inv(Σ_{o,o}) @ (X̃_o − μ̃_o)
        then inverts the transform to return the raw-space value.

        Args:
            Z_prev: shape (T, d_Z).
            P: shape (T, d_P).
            E: shape (T, d_E).
            X_partial: X with held_out component set to NaN; shape (T, d_X).
            spec: ConditioningSpec; all modes must be "fixed".

        Returns:
            Array of shape (T,), raw-space conditional mean of held-out component.

        Raises:
            NotImplementedError: if any mode in spec is not "fixed".
            ValueError: if spec.held_out is not in X_partial.names.
        """
        self._require_fit("inverse")

        for attr, val in [
            ("p_mode", spec.p_mode),
            ("e_mode", spec.e_mode),
            ("z_mode", spec.z_mode),
            ("x_components", spec.x_components),
        ]:
            if val != "fixed":
                raise NotImplementedError(
                    f"RiegelScoreHRStep.inverse: spec.{attr}='{val}' is not supported. "
                    "Only 'fixed' is implemented."
                )

        names = X_partial.names
        if spec.held_out not in names:
            raise ValueError(
                f"spec.held_out='{spec.held_out}' not in X_partial.names {names}"
            )

        h = list(names).index(spec.held_out)
        o_idx = [i for i in range(len(names)) if i != h]

        mu_tilde = self._mean_tilde(Z_prev, P, E)   # (T, d_X)
        x_tilde_partial = _transform(X_partial.values, names)  # (T, d_X); col h may be NaN or log(NaN)

        mu_h = mu_tilde[:, h]                        # (T,)
        mu_o = mu_tilde[:, o_idx]                    # (T, d_X-1)
        x_tilde_o = x_tilde_partial[:, o_idx]        # (T, d_X-1)

        Sigma_ho = self.Σ[h, o_idx]                  # (d_X-1,)
        Sigma_oo = self.Σ[np.ix_(o_idx, o_idx)]      # (d_X-1, d_X-1)
        Sigma_oo_inv = np.linalg.inv(Sigma_oo)        # (d_X-1, d_X-1)

        # μ̃_{h|o} = μ̃_h + Σ_{h,o} @ Σ_{o,o}^{-1} @ (x̃_o − μ̃_o), per row.
        diff = x_tilde_o - mu_o                      # (T, d_X-1)
        correction = diff @ (Sigma_oo_inv @ Sigma_ho)  # (T,)
        mu_h_given_o = mu_h + correction              # (T,)

        return _inverse_transform_component(mu_h_given_o, spec.held_out)

    def log_prob(self, Z_prev: Z, P: P, E: E, X: X) -> Array:
        """Per-row log-density under the fitted Gaussian on X̃.

        Rest-day rows return 0.0; they are not in the observation likelihood
        (handled by the rest transition).

        Args:
            Z_prev: shape (T, d_Z).
            P: shape (T, d_P).
            E: shape (T, d_E).
            X: shape (T, d_X); rest rows have NaN values.

        Returns:
            Array of shape (T,). Rest rows: 0.0. Non-rest rows: log N(X̃; μ̃, Σ).
        """
        self._require_fit("log_prob")
        T = X.values.shape[0]
        result = np.zeros(T, dtype=float)

        active = ~X.is_rest
        if not np.any(active):
            return result

        mu_tilde = self._mean_tilde(Z_prev, P, E)   # (T, d_X)
        x_tilde = _transform(X.values, X.names)      # (T, d_X)

        mu_active = mu_tilde[active]                  # (N, d_X)
        x_active = x_tilde[active]                    # (N, d_X)

        rv = multivariate_normal(mean=np.zeros(x_active.shape[1]), cov=self.Σ, allow_singular=False)
        result[active] = rv.logpdf(x_active - mu_active)

        return result
