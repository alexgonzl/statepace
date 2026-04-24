"""State transition protocols f (workout) and g (rest) for the DAG dynamics (architecture_map §3.3)."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from scipy.stats import multivariate_normal
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


# ---------------------------------------------------------------------------
# Reference implementation: LinearGaussian transitions
# ---------------------------------------------------------------------------

# π_stim component names (riegel-score-hrstep paired impl, §Channel composition).
_PI_STIM_NAMES: tuple[str, ...] = (
    "hr_load",
    "step_load",
    "total_elevation_gain",
    "total_elevation_lost",
    "heat_exposure",
)
_N_STIM = len(_PI_STIM_NAMES)   # 5


class LinearGaussianWorkoutTransition:
    """Reference WorkoutTransition: Z_t = F·Z_{t-1} + G·π_stim + m + ν, ν~N(0,Q).

    See docs/reference_impls/linear-gaussian.md §Family (workout transition).
    """

    def __init__(self, d_Z: int) -> None:
        self.d_Z = d_Z
        self.F: Array | None = None   # (d_Z, d_Z)
        self.G: Array | None = None   # (d_Z, 5)
        self.m: Array | None = None   # (d_Z,)
        self.Q: Array | None = None   # (d_Z, d_Z)

    def _require_fit(self, method: str) -> None:
        """Raise RuntimeError if the model has not been fit.

        Args:
            method: name of the calling method, included in the error message.
        """
        if self.F is None:
            raise RuntimeError(
                f"LinearGaussianWorkoutTransition.{method} called before fit(). "
                "Call fit(Z_prev, X_t, Z_t) first."
            )

    def fit(self, Z_prev: Z, X_t: X, Z_t: Z) -> "LinearGaussianWorkoutTransition":
        """Fit F, G, m, Q by multivariate OLS on workout-day rows.

        Select rows where X_t.is_rest == False. Build
        Φ = [Z_prev.mean | π_stim(X_t) | 1] where π_stim reads the 5
        load components from X_t by name. OLS via np.linalg.lstsq.
        Partition Θ into F, G, m; compute Q as residual sample covariance.
        Returns a new fitted instance; does not mutate self.

        Args:
            Z_prev: latent state at t-1, shape (T, d_Z).
            X_t: execution channels at t, shape (T, d_X); rest rows NaN.
            Z_t: latent state at t, shape (T, d_Z).

        Returns:
            New fitted LinearGaussianWorkoutTransition.

        Raises:
            ValueError: if Z_prev.mean.shape[1] != self.d_Z.
            ValueError: if any of the 5 π_stim names are missing from X_t.names.
            RuntimeError: if no workout-day rows exist after filtering.
        """
        if Z_prev.mean.shape[1] != self.d_Z:
            raise ValueError(
                f"Z_prev.mean has d_Z={Z_prev.mean.shape[1]}, "
                f"but LinearGaussianWorkoutTransition was constructed with d_Z={self.d_Z}."
            )

        missing = [n for n in _PI_STIM_NAMES if n not in X_t.names]
        if missing:
            raise ValueError(
                f"π_stim names missing from X_t.names: {missing}"
            )

        active = ~X_t.is_rest   # (T,) bool
        if not np.any(active):
            raise RuntimeError("No workout-day rows found after filtering on X_t.is_rest.")

        z_prev_act = Z_prev.mean[active]    # (N, d_Z)
        z_t_act = Z_t.mean[active]          # (N, d_Z)

        # Extract π_stim columns by name.
        name_to_col = {name: i for i, name in enumerate(X_t.names)}
        stim_idx = [name_to_col[n] for n in _PI_STIM_NAMES]
        pi_stim = X_t.values[active][:, stim_idx]   # (N, 5)

        N = z_prev_act.shape[0]
        ones = np.ones((N, 1), dtype=float)
        Phi = np.concatenate([z_prev_act, pi_stim, ones], axis=1)  # (N, d_Z+6)

        Theta, _, _, _ = np.linalg.lstsq(Phi, z_t_act, rcond=None)  # (d_Z+6, d_Z)

        F = Theta[:self.d_Z, :].T                        # (d_Z, d_Z)
        G = Theta[self.d_Z: self.d_Z + _N_STIM, :].T    # (d_Z, 5)
        m = Theta[self.d_Z + _N_STIM, :]                 # (d_Z,)

        residuals = z_t_act - Phi @ Theta                # (N, d_Z)
        n_params = Phi.shape[1]
        Q = (residuals.T @ residuals) / (N - n_params)   # (d_Z, d_Z)

        inst = self.__class__(d_Z=self.d_Z)
        inst.F = F
        inst.G = G
        inst.m = m
        inst.Q = Q
        return inst

    def _compute_mean(self, Z_prev: Z, X_t: X) -> Array:
        """Compute conditional mean F·Z_{t-1} + G·π_stim + m for all rows.

        Args:
            Z_prev: shape (T, d_Z).
            X_t: shape (T, d_X).

        Returns:
            Array of shape (T, d_Z).
        """
        name_to_col = {name: i for i, name in enumerate(X_t.names)}
        stim_idx = [name_to_col[n] for n in _PI_STIM_NAMES]
        pi_stim = X_t.values[:, stim_idx]    # (T, 5)

        return Z_prev.mean @ self.F.T + pi_stim @ self.G.T + self.m  # (T, d_Z)

    def step(self, Z_prev: Z, X_t: X) -> Z:
        """Return conditional mean Z_t = F·Z_{t-1} + G·π_stim + m on all rows.

        Does not add noise; the Protocol is silent on sampling, mean is canonical.

        Args:
            Z_prev: shape (T, d_Z).
            X_t: shape (T, d_X).

        Returns:
            Z with mean shape (T, d_Z), cov=None, dates copied from Z_prev.
        """
        self._require_fit("step")
        mean = self._compute_mean(Z_prev, X_t)
        return Z(mean=mean, cov=None, dates=Z_prev.dates)

    def log_prob(self, Z_prev: Z, X_t: X, Z_t: Z) -> Array:
        """Per-row log N(Z_t; F·Z_{t-1} + G·π_stim + m, Q). Rest rows return 0.0.

        Args:
            Z_prev: shape (T, d_Z).
            X_t: shape (T, d_X).
            Z_t: shape (T, d_Z).

        Returns:
            Array of shape (T,). Rest rows: 0.0. Non-rest rows: multivariate Gaussian log-density.
        """
        self._require_fit("log_prob")
        T = Z_t.mean.shape[0]
        result = np.zeros(T, dtype=float)

        active = ~X_t.is_rest
        if not np.any(active):
            return result

        mu = self._compute_mean(Z_prev, X_t)   # (T, d_Z)

        rv = multivariate_normal(
            mean=np.zeros(self.d_Z), cov=self.Q, allow_singular=False
        )
        result[active] = rv.logpdf(Z_t.mean[active] - mu[active])

        return result


class LinearGaussianRestTransition:
    """Reference RestTransition: Z_t = H^n·Z_{t-1} + r(n) + ν, ν~N(0, Q_rest(n)).

    See docs/reference_impls/linear-gaussian.md §Family (rest transition).
    """

    def __init__(
        self,
        d_Z: int,
        max_consecutive_rest_days: int,
        eig_eps: float = 1e-4,
    ) -> None:
        self.d_Z = d_Z
        self.max_consecutive_rest_days = max_consecutive_rest_days
        self.eig_eps = eig_eps
        self.H: Array | None = None    # (d_Z, d_Z)
        self.r_1: Array | None = None  # (d_Z,)
        self.Q_1: Array | None = None  # (d_Z, d_Z)

    def _require_fit(self, method: str) -> None:
        """Raise RuntimeError if the model has not been fit.

        Args:
            method: name of the calling method, included in the error message.
        """
        if self.H is None:
            raise RuntimeError(
                f"LinearGaussianRestTransition.{method} called before fit(). "
                "Call fit(Z_prev, n_rest_days, Z_t) first."
            )

    def fit(
        self, Z_prev: Z, n_rest_days: Array, Z_t: Z
    ) -> "LinearGaussianRestTransition":
        """Fit H, r_1, Q_1 by OLS on n=1 rest-day rows, then project H.

        Select rows where n_rest_days == 1. Build Φ = [Z_prev.mean | 1]; OLS
        gives H_raw (unconstrained) and r_1. Project H onto the feasible set:
          1. Symmetrize: H_sym = (H_raw + H_raw.T) / 2.
          2. Eigendecompose H_sym = U Λ U.T.
          3. Clip eigenvalues to [eig_eps, 1 - eig_eps].
          4. Reconstruct H = U Λ_clipped U.T.
        Q_1 is the sample residual covariance from the n=1 OLS.

        Returns a new fitted instance; does not mutate self.

        Args:
            Z_prev: latent state at t-1, shape (T, d_Z).
            n_rest_days: integer array shape (T,); value at each row is the
                count of consecutive rest days ending at that day (1-based).
            Z_t: latent state at t, shape (T, d_Z).

        Returns:
            New fitted LinearGaussianRestTransition.

        Raises:
            ValueError: if Z_prev.mean.shape[1] != self.d_Z.
            RuntimeError: if no n=1 rest-day rows exist.
        """
        if Z_prev.mean.shape[1] != self.d_Z:
            raise ValueError(
                f"Z_prev.mean has d_Z={Z_prev.mean.shape[1]}, "
                f"but LinearGaussianRestTransition was constructed with d_Z={self.d_Z}."
            )

        mask = (n_rest_days == 1)
        if not np.any(mask):
            raise RuntimeError("No n=1 rest-day rows found in n_rest_days.")

        z_prev_act = Z_prev.mean[mask]   # (N, d_Z)
        z_t_act = Z_t.mean[mask]         # (N, d_Z)

        N = z_prev_act.shape[0]
        ones = np.ones((N, 1), dtype=float)
        Phi = np.concatenate([z_prev_act, ones], axis=1)   # (N, d_Z+1)

        Theta, _, _, _ = np.linalg.lstsq(Phi, z_t_act, rcond=None)  # (d_Z+1, d_Z)

        H_raw = Theta[:self.d_Z, :].T   # (d_Z, d_Z)
        r_1 = Theta[self.d_Z, :]        # (d_Z,)

        # Project H_raw onto symmetric-PSD with eigenvalues in [eig_eps, 1-eig_eps].
        H_sym = (H_raw + H_raw.T) / 2.0
        eigvals, U = np.linalg.eigh(H_sym)
        eigvals_clipped = np.clip(eigvals, self.eig_eps, 1.0 - self.eig_eps)
        H = U @ np.diag(eigvals_clipped) @ U.T

        residuals = z_t_act - Phi @ Theta              # (N, d_Z)
        n_params = Phi.shape[1]
        Q_1 = (residuals.T @ residuals) / (N - n_params)   # (d_Z, d_Z)

        inst = self.__class__(
            d_Z=self.d_Z,
            max_consecutive_rest_days=self.max_consecutive_rest_days,
            eig_eps=self.eig_eps,
        )
        inst.H = H
        inst.r_1 = r_1
        inst.Q_1 = Q_1
        return inst

    def _H_power(self, n: int) -> Array:
        """Compute H^n via eigendecomposition for numerical stability.

        Args:
            n: non-negative integer exponent.

        Returns:
            Array of shape (d_Z, d_Z).
        """
        eigvals, U = np.linalg.eigh(self.H)
        return U @ np.diag(eigvals ** n) @ U.T

    def _r_n(self, n: int) -> Array:
        """Compute r(n) via the recursion r(k) = H·r(k-1) + r_1, r(0)=0.

        Args:
            n: number of rest days (>= 1).

        Returns:
            Array of shape (d_Z,).
        """
        r = np.zeros(self.d_Z, dtype=float)
        for _ in range(n):
            r = self.H @ r + self.r_1
        return r

    def _Q_rest_n(self, n: int) -> Array:
        """Compute Q_rest(n) via recursion Q(k) = H·Q(k-1)·H.T + Q_1, Q(0)=0.

        Args:
            n: number of rest days (>= 1).

        Returns:
            Array of shape (d_Z, d_Z).
        """
        Q = np.zeros((self.d_Z, self.d_Z), dtype=float)
        for _ in range(n):
            Q = self.H @ Q @ self.H.T + self.Q_1
        return Q

    def _validate_n(self, n: int) -> None:
        """Raise ValueError if n is out of the valid range [1, max_consecutive_rest_days].

        Args:
            n: consecutive rest day count.
        """
        if n < 1:
            raise ValueError(
                f"n_rest_days={n} < 1; rest-day count must be at least 1."
            )
        if n > self.max_consecutive_rest_days:
            raise ValueError(
                f"n_rest_days={n} > max_consecutive_rest_days="
                f"{self.max_consecutive_rest_days}; Z is undefined past the bound (A5)."
            )

    def step(self, Z_prev: Z, n_rest_days: int) -> Z:
        """Return conditional mean Z_t = H^n · Z_{t-1} + r(n).

        r(n) computed via the numerically stable recursion r(k) = H·r(k-1) + r_1,
        r(0) = 0. H^n computed via eigendecomposition for stability at large n.

        Args:
            Z_prev: latent state, shape (T, d_Z).
            n_rest_days: scalar int; number of consecutive rest days.

        Returns:
            Z with mean shape (T, d_Z), cov=None, dates copied from Z_prev.

        Raises:
            ValueError: if n_rest_days < 1 or > self.max_consecutive_rest_days.
        """
        self._require_fit("step")
        self._validate_n(n_rest_days)

        H_n = self._H_power(n_rest_days)   # (d_Z, d_Z)
        r_n = self._r_n(n_rest_days)       # (d_Z,)

        mean = Z_prev.mean @ H_n.T + r_n   # (T, d_Z)
        return Z(mean=mean, cov=None, dates=Z_prev.dates)

    def log_prob(self, Z_prev: Z, n_rest_days: int, Z_t: Z) -> Array:
        """Per-row log N(Z_t; H^n·Z_{t-1} + r(n), Q_rest(n)).

        Q_rest(n) computed via recursion Q(k) = H·Q(k-1)·H.T + Q_1, Q(0) = 0.

        Args:
            Z_prev: shape (T, d_Z).
            n_rest_days: scalar int.
            Z_t: shape (T, d_Z).

        Returns:
            Array of shape (T,), per-row log-density.
        """
        self._require_fit("log_prob")
        self._validate_n(n_rest_days)

        H_n = self._H_power(n_rest_days)    # (d_Z, d_Z)
        r_n = self._r_n(n_rest_days)        # (d_Z,)
        Q_n = self._Q_rest_n(n_rest_days)   # (d_Z, d_Z)

        mu = Z_prev.mean @ H_n.T + r_n     # (T, d_Z)
        rv = multivariate_normal(
            mean=np.zeros(self.d_Z), cov=Q_n, allow_singular=False
        )
        return rv.logpdf(Z_t.mean - mu)
