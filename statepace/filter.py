"""State inference protocol and Prior container for p(Z_t | history) (architecture_map §3.4).

Also contains: ZPosterior sealed ABC, GaussianZPosterior concrete subclass, and
JointMLEKalman reference StateEstimator implementation.

Spec: docs/reference_impls/joint-mle-kalman.md
ADR:  docs/decisions/0006-m6-joint-mle-kalman-first-state-estimator.md
"""
from __future__ import annotations

import os
# Required on macOS when torch and numpy both link libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Literal, Mapping, Sequence
import numpy as np

from statepace.channels import Channels, Z, Array
from statepace.observation import ObservationModel
from statepace.transitions import WorkoutTransition, RestTransition

InferMode = Literal["filter", "smooth"]


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ZPosterior sealed ABC + GaussianZPosterior
# ---------------------------------------------------------------------------

class ZPosterior(ABC):
    """Sealed ABC for the filtered / smoothed Z posterior (spec §infer Protocol widening).

    Successor estimator families (variational, particle, switching SSM) define
    their own concrete subclasses without re-widening this Protocol.
    M6 returns GaussianZPosterior. M7+ forward.py and predict.py consume
    the ABC interface.

    Sealing: subclassing outside statepace is not prevented at runtime, but
    the ABC is declared internal; the public contract is the three abstract
    methods below.

    Attributes:
        dates: datetime64[ns] array, shape (T,).
        d_Z: latent dimensionality.
    """

    dates: Array   # datetime64[ns], shape (T,)
    d_Z: int

    @abstractmethod
    def mean(self) -> Array:
        """Posterior mean trajectory.

        Returns:
            Array of shape (T, d_Z).
        """
        ...

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> Array:
        """Draw n sample trajectories from the posterior.

        Args:
            n: number of samples.
            rng: numpy random Generator for reproducibility.

        Returns:
            Array of shape (n, T, d_Z).
        """
        ...

    @abstractmethod
    def marginal_log_pdf(self, z: Array) -> Array:
        """Per-timestep marginal log-density evaluated at z.

        Args:
            z: Array of shape (T, d_Z); one state per timestep.

        Returns:
            Array of shape (T,) of log-densities.
        """
        ...


@dataclass(frozen=True)
class GaussianZPosterior(ZPosterior):
    """Gaussian Kalman-filter posterior over Z (M6 concrete subclass).

    Holds the per-timestep marginal mean and covariance from the forward
    Kalman pass. Returned by JointMLEKalman.infer.

    Args:
        _mean: shape (T, d_Z), posterior mean trajectory.
        cov: shape (T, d_Z, d_Z), posterior marginal covariances.
        dates: datetime64[ns], shape (T,).
    """
    _mean: Array          # (T, d_Z)
    cov: Array            # (T, d_Z, d_Z)
    dates: Array          # datetime64[ns], shape (T,)

    @property
    def d_Z(self) -> int:
        """Latent dimensionality inferred from stored mean."""
        return self._mean.shape[1]

    def mean(self) -> Array:
        """Posterior mean trajectory.

        Returns:
            Array of shape (T, d_Z).
        """
        return self._mean

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        """Draw n trajectory samples using per-timestep marginal Gaussians.

        Note: samples are drawn from marginals independently (not from the
        joint trajectory distribution). Joint-trajectory sampling requires
        the RTS smoother (M7+).

        Args:
            n: number of samples.
            rng: numpy random Generator.

        Returns:
            Array of shape (n, T, d_Z).
        """
        T, d_Z = self._mean.shape
        out = np.empty((n, T, d_Z), dtype=float)
        for t in range(T):
            out[:, t, :] = rng.multivariate_normal(
                mean=self._mean[t], cov=self.cov[t], size=n
            )
        return out

    def marginal_log_pdf(self, z: Array) -> Array:
        """Per-timestep log N(z_t; μ_t, Σ_t).

        Args:
            z: Array of shape (T, d_Z).

        Returns:
            Array of shape (T,).
        """
        from scipy.stats import multivariate_normal as mvn

        T, d_Z = self._mean.shape
        out = np.empty(T, dtype=float)
        for t in range(T):
            out[t] = mvn.logpdf(z[t], mean=self._mean[t], cov=self.cov[t])
        return out


# ---------------------------------------------------------------------------
# StateEstimator Protocol
# ---------------------------------------------------------------------------

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
    ) -> ZPosterior: ...


# ---------------------------------------------------------------------------
# JointMLEKalman config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointMLEKalmanConfig:
    """Hyperparameters for JointMLEKalman.

    Defaults reflect W-D3–D13 decisions in spec §Hyperparameters and ADR 0006.
    Override conservatively; the synthetic-recovery test drives defaults below.

    Attributes:
        d_Z: latent dimensionality; must match paired impls.
        tau: fixed time constants (days), one per Z dimension. Length must equal d_Z.
        n_pi_stim: number of stimulus channels (G shape is (d_Z, n_pi_stim)).
        learning_rate: Adam initial learning rate (W-D3).
        min_learning_rate: cosine-decay floor.
        max_iterations: iteration cap (W-D3). Chosen so synthetic recovery passes.
        patience: validation-NLL plateau patience in eval_every units (W-D5).
        eval_every: epoch cadence for validation-NLL evaluation (W-D5).
        tol_conv: relative-improvement tolerance for convergence (W-D5).
        n_seeds: multi-start count (W-D4).
        q_init: init magnitude for Q, Q_1 diagonal entries (W-D7).
        sigma0_sq: diffuse prior variance for Z_0 (W-D7).
        tol_b: sign-cascade primary tolerance (W-D13).
        tol_m: sign-cascade fallback tolerance (W-D13).
        sigma_re_sq: rest-bound re-entry prior variance (W-D10).
        val_fraction: fraction of cohort held out for validation during fit.
        infer_max_iterations: iteration cap for validation-cohort mu_0 fitting.
    """
    d_Z: int = 4
    tau: tuple[float, ...] = (1.0, 7.0, 28.0, 84.0)
    n_pi_stim: int = 5
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    max_iterations: int = 2000
    patience: int = 20
    eval_every: int = 10
    tol_conv: float = 1e-4
    n_seeds: int = 5
    q_init: float = 0.01
    sigma0_sq: float = 100.0
    tol_b: float = 0.05
    tol_m: float = 0.05
    sigma_re_sq: float = 1.0
    val_fraction: float = 0.2
    infer_max_iterations: int = 500


# ---------------------------------------------------------------------------
# JointMLEKalman implementation
# ---------------------------------------------------------------------------

# π_stim component names — must match the paired riegel-score-hrstep impl.
_PI_STIM_NAMES: tuple[str, ...] = (
    "hr_load",
    "step_load",
    "total_elevation_gain",
    "total_elevation_lost",
    "heat_exposure",
)


class JointMLEKalman:
    """Reference StateEstimator: joint MLE via differentiable Kalman filter (spec §joint-mle-kalman).

    Hyperparameter defaults (all W-Dn decisions from spec §Hyperparameters):
        learning_rate = 1e-3 (Adam, cosine decay to 1e-5, W-D3)
        max_iterations = 2000 (W-D3; conservative; synthetic recovery uses ~500 iters)
        patience = 20 evaluations × eval_every=10 = 200 epochs (W-D5)
        eval_every = 10 epochs (W-D5)
        tol_conv = 1e-4 relative (W-D5)
        n_seeds = 5 (W-D4)
        q_init = 0.01 (W-D7)
        sigma0_sq = 100.0 (W-D7, diffuse)
        tol_b = 0.05, tol_m = 0.05 (W-D13)
        sigma_re_sq = 1.0 (W-D10; re-entry prior variance around detraining target b)
        val_fraction = 0.2 (20% of cohort held out for convergence monitoring)
        infer_max_iterations = 500 (validation-cohort mu_0 fitting cap)

    Structural priors baked in (cannot be escaped by optimizer):
        H fully diagonal: H_ii = exp(-1/tau_i)
        F diagonal fixed: F_ii = exp(-1/tau_i); off-diagonals F_ij = alpha_ij*(1-F_ii)
        r_1 derived: r_1 = (I - H) @ b
        G_ij = (1 - F_ii) * g_ij (bounded per-row)
        Cholesky-PSD covariances
        pi_stim cohort-mean centering, frozen at fit time
    """

    def __init__(self, cfg: JointMLEKalmanConfig | None = None) -> None:
        """Initialize JointMLEKalman with optional config.

        Args:
            cfg: JointMLEKalmanConfig; uses defaults if None.
        """
        self.cfg = cfg if cfg is not None else JointMLEKalmanConfig()
        self.d_Z = self.cfg.d_Z

        # Populated after fit():
        self._theta: dict | None = None          # shared parameters (numpy)
        self._mu0: dict[str, Array] | None = None  # per-athlete mu_0^{(i)}
        self._pi_bar_stim: Array | None = None   # frozen cohort-mean pi_stim
        self._x_names: tuple[str, ...] | None = None
        self._p_names: tuple[str, ...] | None = None
        self._e_names: tuple[str, ...] | None = None
        self._train_subject_ids: list[str] | None = None
        self._max_rest_days: int | None = None   # from rest_transition.max_consecutive_rest_days
        self.gauge_boundary_dims: list[int] = []
        self.selection_log_lik: float | None = None
        self.multi_start_diagnostics: list[dict] | None = None

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        cohort: Mapping[str, Channels],
        observation: ObservationModel,
        workout_transition: WorkoutTransition,
        rest_transition: RestTransition,
        prior: Prior | Mapping[str, Prior],
    ) -> "JointMLEKalman":
        """Fit all shared parameters and per-athlete mu_0^{(i)} via joint SGD.

        Construction-time error if d_Z mismatches any paired impl. Returns a
        new fitted instance; does not mutate self.

        Args:
            cohort: mapping subject_id -> Channels.
            observation: ObservationModel with matching d_Z.
            workout_transition: WorkoutTransition with matching d_Z.
            rest_transition: RestTransition with matching d_Z.
            prior: Prior or per-athlete mapping. If Prior.diffuse=True, sigma0_sq
                is used regardless of prior.cov.

        Returns:
            New fitted JointMLEKalman instance.

        Raises:
            ValueError: if d_Z mismatches any paired impl.
        """
        import torch

        cfg = self.cfg
        _check_dz(cfg.d_Z, observation, workout_transition, rest_transition)

        max_rest_days = rest_transition.max_consecutive_rest_days

        subject_ids = list(cohort.keys())
        n_athletes = len(subject_ids)

        # Split cohort into train/val.
        rng_split = np.random.default_rng(0)
        n_val = max(1, int(n_athletes * cfg.val_fraction))
        val_idx = set(rng_split.choice(n_athletes, size=n_val, replace=False).tolist())
        train_ids = [sid for i, sid in enumerate(subject_ids) if i not in val_idx]
        val_ids = [sid for i, sid in enumerate(subject_ids) if i in val_idx]

        if len(train_ids) == 0:
            train_ids = subject_ids
            val_ids = subject_ids

        # Determine channel names from first athlete.
        first_ch = next(iter(cohort.values()))
        x_names = first_ch.X.names
        p_names = first_ch.P.names
        e_names = first_ch.E.names

        # Compute pi_bar_stim from training cohort (cohort-mean over workout days).
        pi_bar_stim = _compute_pi_bar_stim(
            {sid: cohort[sid] for sid in train_ids}, x_names
        )

        # Staged init (deterministic given cohort data).
        theta_init = _staged_init(
            cohort={sid: cohort[sid] for sid in train_ids},
            x_names=x_names,
            p_names=p_names,
            e_names=e_names,
            pi_bar_stim=pi_bar_stim,
            cfg=cfg,
        )

        # Multi-start SGD.
        best_val_ll = -np.inf
        best_result = None
        diagnostics: list[dict] = []

        for seed_idx in range(cfg.n_seeds):
            seed = 1000 + seed_idx
            theta_s = _perturb_theta(theta_init, seed=seed, cfg=cfg)
            mu0_s = {sid: np.zeros(cfg.d_Z, dtype=float) for sid in train_ids}

            theta_final, mu0_final, diag = _sgd_fit(
                theta=theta_s,
                mu0_train=mu0_s,
                train_ids=train_ids,
                val_ids=val_ids,
                cohort=cohort,
                x_names=x_names,
                p_names=p_names,
                e_names=e_names,
                pi_bar_stim=pi_bar_stim,
                cfg=cfg,
                seed=seed,
                max_rest_days=max_rest_days,
            )
            val_ll = diag["val_log_lik"]
            diag["seed"] = seed_idx
            diag["selected"] = False
            diagnostics.append(diag)

            if best_result is None or val_ll > best_val_ll:
                best_val_ll = val_ll
                best_result = (theta_final, mu0_final)

        # Mark selected seed.
        best_seed_idx = int(np.argmax([d["val_log_lik"] for d in diagnostics]))
        diagnostics[best_seed_idx]["selected"] = True

        theta_best, mu0_best = best_result
        theta_best = _apply_sign_cascade(theta_best, cfg)
        gauge_dims = _find_gauge_boundary_dims(theta_best, cfg)

        # Build fitted instance.
        inst = JointMLEKalman(cfg=cfg)
        inst._theta = theta_best
        inst._mu0 = mu0_best
        inst._pi_bar_stim = pi_bar_stim
        inst._x_names = x_names
        inst._p_names = p_names
        inst._e_names = e_names
        inst._train_subject_ids = train_ids
        inst._max_rest_days = max_rest_days
        inst.gauge_boundary_dims = gauge_dims
        inst.selection_log_lik = float(best_val_ll)
        inst.multi_start_diagnostics = diagnostics

        return inst

    # ------------------------------------------------------------------
    # Public: infer
    # ------------------------------------------------------------------

    def infer(
        self,
        channels: Channels,
        mode: InferMode = "filter",
        prior: Prior | None = None,
    ) -> ZPosterior:
        """Run Kalman filter and return GaussianZPosterior.

        For training-cohort athletes: uses frozen mu_0^{(i)} from fit.
        For validation-cohort athletes: fits mu_0^{(i)} by SGD under frozen theta,
        then runs the filter.

        Args:
            channels: Channels for one athlete.
            mode: "filter" only at M6; "smooth" raises NotImplementedError.
            prior: if provided and diffuse=True, uses large sigma0_sq prior.

        Returns:
            GaussianZPosterior with mean (T, d_Z) and cov (T, d_Z, d_Z).

        Raises:
            NotImplementedError: if mode == "smooth".
            RuntimeError: if called before fit().
        """
        if mode == "smooth":
            raise NotImplementedError(
                "JointMLEKalman: mode='smooth' deferred to M7+ (W-D8 closed in spec). "
                "Use mode='filter'."
            )
        if self._theta is None:
            raise RuntimeError(
                "JointMLEKalman.infer() called before fit(). Call fit() first."
            )

        sid = channels.subject_id

        # Determine mu_0 for this athlete.
        if sid in (self._train_subject_ids or []) and self._mu0 is not None and sid in self._mu0:
            mu0 = self._mu0[sid]
        else:
            # Validation-cohort: fit mu_0 under frozen theta.
            mu0 = _fit_mu0_single(
                theta=self._theta,
                channels=channels,
                x_names=self._x_names,
                p_names=self._p_names,
                e_names=self._e_names,
                pi_bar_stim=self._pi_bar_stim,
                cfg=self.cfg,
                max_rest_days=self._max_rest_days if self._max_rest_days is not None else 10,
            )

        # Run Kalman filter forward.
        mean_traj, cov_traj = _kalman_filter_numpy(
            theta=self._theta,
            mu0=mu0,
            channels=channels,
            x_names=self._x_names,
            p_names=self._p_names,
            e_names=self._e_names,
            pi_bar_stim=self._pi_bar_stim,
            cfg=self.cfg,
            max_rest_days=self._max_rest_days if self._max_rest_days is not None else 10,
        )

        return GaussianZPosterior(
            _mean=mean_traj,
            cov=cov_traj,
            dates=channels.dates.copy(),
        )


# ---------------------------------------------------------------------------
# Internal: d_Z check
# ---------------------------------------------------------------------------

def _check_dz(
    d_Z: int,
    observation: ObservationModel,
    workout_transition: WorkoutTransition,
    rest_transition: RestTransition,
) -> None:
    """Raise ValueError if any paired impl has a mismatched d_Z.

    Args:
        d_Z: expected latent dimensionality.
        observation: ObservationModel.
        workout_transition: WorkoutTransition.
        rest_transition: RestTransition.
    """
    for impl, name in [
        (observation, "observation"),
        (workout_transition, "workout_transition"),
        (rest_transition, "rest_transition"),
    ]:
        if hasattr(impl, "d_Z") and impl.d_Z != d_Z:
            raise ValueError(
                f"JointMLEKalman d_Z={d_Z} but {name}.d_Z={impl.d_Z}. "
                "All paired impls must have matching d_Z."
            )


# ---------------------------------------------------------------------------
# Internal: pi_stim utilities
# ---------------------------------------------------------------------------

def _get_pi_stim(x_values: Array, x_names: tuple[str, ...]) -> Array:
    """Extract the 5 pi_stim components from X by name.

    Args:
        x_values: shape (T, d_X).
        x_names: component names aligned with columns.

    Returns:
        Array of shape (T, 5).
    """
    name_to_col = {n: i for i, n in enumerate(x_names)}
    idx = [name_to_col[n] for n in _PI_STIM_NAMES]
    return x_values[:, idx]


def _compute_pi_bar_stim(
    cohort: Mapping[str, Channels],
    x_names: tuple[str, ...],
) -> Array:
    """Compute cohort-mean pi_stim over all training-cohort workout days.

    Args:
        cohort: training cohort.
        x_names: X component names.

    Returns:
        Array of shape (5,), cohort-mean pi_stim.
    """
    parts = []
    for ch in cohort.values():
        active = ~ch.X.is_rest
        if np.any(active):
            pi_stim = _get_pi_stim(ch.X.values[active], x_names)
            parts.append(pi_stim)
    if not parts:
        return np.zeros(len(_PI_STIM_NAMES), dtype=float)
    all_pi = np.concatenate(parts, axis=0)
    return all_pi.mean(axis=0)


# ---------------------------------------------------------------------------
# Internal: structural parameter construction
# ---------------------------------------------------------------------------

def _build_F_H(cfg: JointMLEKalmanConfig) -> tuple[Array, Array]:
    """Build fixed diagonal entries of F and H from tau schedule.

    F_ii = H_ii = exp(-1/tau_i).

    Args:
        cfg: config with tau and d_Z.

    Returns:
        (f_diag, h_diag): arrays of shape (d_Z,).
    """
    tau = np.array(cfg.tau, dtype=float)
    diag = np.exp(-1.0 / tau)
    return diag.copy(), diag.copy()


def _theta_to_arrays(theta: dict, cfg: JointMLEKalmanConfig) -> dict[str, Array]:
    """Convert theta dict (numpy float64) to named arrays.

    Enforces structural parameterization:
        F_ii = exp(-1/tau_i) fixed; F_ij = tanh(beta_ij) * (1 - F_ii)
        H diagonal: H_ii = exp(-1/tau_i)
        G_ij = (1 - F_ii) * g_ij
        r_1 = (I - H) @ b
        Sigma = L_Sigma @ L_Sigma.T  (L_Sigma lower triangular, diag via softplus)
        Q = L_Q @ L_Q.T
        Q_1 = L_Q1 @ L_Q1.T

    Args:
        theta: parameter dict from _sgd_fit or _staged_init.
        cfg: config.

    Returns:
        Dict with keys: F, H, G, b, r_1, m, Sigma, Q, Q_1, A, B, C, d_vec.
        All arrays are float64 numpy.
    """
    dZ = cfg.d_Z
    f_diag, h_diag = _build_F_H(cfg)

    # F: diagonal fixed, off-diagonals from beta
    beta = theta["beta"]              # (d_Z, d_Z) — diagonal entries ignored
    alpha = np.tanh(beta)             # (d_Z, d_Z)
    F = np.diag(f_diag).copy()
    for i in range(dZ):
        for j in range(dZ):
            if i != j:
                F[i, j] = alpha[i, j] * (1.0 - f_diag[i])

    # H diagonal (no free params)
    H = np.diag(h_diag)

    # G_ij = (1 - F_ii) * g_ij
    g_raw = theta["g_raw"]            # (d_Z, n_pi_stim)
    G = g_raw * (1.0 - f_diag)[:, np.newaxis]

    # b (cohort-shared detraining target), m (workout drift)
    b = theta["b"].copy()
    m = theta["m"].copy()

    # r_1 derived from b and H
    r_1 = (np.eye(dZ) - H) @ b

    # Covariances via Cholesky
    Sigma = _chol_param_to_cov(theta["L_Sigma_raw"], dZ)
    Q = _chol_param_to_cov(theta["L_Q_raw"], dZ)
    Q_1 = _chol_param_to_cov(theta["L_Q1_raw"], dZ)

    # Observation coefficients
    A = theta["A"].copy()             # (d_X, d_Z)
    B = theta["B"].copy()             # (d_X, d_P)
    C = theta["C"].copy()             # (d_X, d_E)
    d_vec = theta["d_vec"].copy()     # (d_X,)

    return dict(F=F, H=H, G=G, b=b, r_1=r_1, m=m,
                Sigma=Sigma, Q=Q, Q_1=Q_1, A=A, B=B, C=C, d_vec=d_vec)


def _chol_param_to_cov(L_raw: Array, d: int) -> Array:
    """Convert unconstrained Cholesky raw parameters to a PSD matrix.

    Lower triangular L: off-diagonals are free; diagonal elements are
    softplus(raw_diag) to ensure positivity.

    Args:
        L_raw: shape (d, d), lower triangular raw parameters.
        d: dimension.

    Returns:
        PSD matrix of shape (d, d).
    """
    L = np.tril(L_raw)
    # Softplus on diagonal to ensure positive.
    for i in range(d):
        L[i, i] = np.log1p(np.exp(L_raw[i, i]))
    return L @ L.T


# PyTorch version for autograd.
def _chol_param_to_cov_torch(L_raw, d: int):
    """Torch version: convert raw Cholesky params to PSD matrix.

    Args:
        L_raw: (d, d) torch tensor, lower triangular.
        d: dimension.

    Returns:
        (d, d) PSD torch tensor.
    """
    import torch
    import torch.nn.functional as F
    L = torch.tril(L_raw)
    # Zero out diagonal from tril, add softplus diagonal.
    diag_mask = torch.eye(d, dtype=L.dtype, device=L.device).bool()
    L = L.masked_fill(diag_mask, 0.0)
    L = L + torch.diag(F.softplus(torch.diag(L_raw)))
    return L @ L.T


# ---------------------------------------------------------------------------
# Internal: staged init
# ---------------------------------------------------------------------------

def _staged_init(
    cohort: Mapping[str, Channels],
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim: Array,
    cfg: JointMLEKalmanConfig,
) -> dict:
    """Compute deterministic staged init (PCA -> OLS -> small identity).

    Deterministic given training cohort. Small-random components (b, m, g_raw,
    Cholesky off-diagonals) are set to zero here and perturbed per seed in
    _perturb_theta.

    Args:
        cohort: training cohort.
        x_names: X component names.
        p_names: P component names.
        e_names: E component names.
        pi_bar_stim: shape (n_pi_stim,), frozen cohort-mean offset.
        cfg: config.

    Returns:
        theta dict with all parameters initialized.
    """
    dZ = cfg.d_Z

    # Collect all workout-day rows for PCA and OLS init.
    x_tilde_list, p_list, e_list, pi_stim_list = [], [], [], []
    for ch in cohort.values():
        active = ~ch.X.is_rest
        if not np.any(active):
            continue
        from statepace.observation import _transform
        x_tilde = _transform(ch.X.values[active], x_names)  # (N, d_X)
        p_vals = ch.P.values[active]
        e_vals = ch.E.values[active]
        pi_stim_centered = _get_pi_stim(ch.X.values[active], x_names) - pi_bar_stim
        x_tilde_list.append(x_tilde)
        p_list.append(p_vals)
        e_list.append(e_vals)
        pi_stim_list.append(pi_stim_centered)

    if not x_tilde_list:
        raise RuntimeError("No workout rows found in training cohort for staged init.")

    X_all = np.concatenate(x_tilde_list, axis=0)   # (N, d_X)
    P_all = np.concatenate(p_list, axis=0)
    E_all = np.concatenate(e_list, axis=0)

    # OLS init: regress X̃ on (P, E, 1) to init B, C, d, and residual Sigma.
    N, d_X = X_all.shape
    d_P = P_all.shape[1]
    d_E = E_all.shape[1]
    ones = np.ones((N, 1), dtype=float)
    Phi_ols = np.concatenate([P_all, E_all, ones], axis=1)   # (N, d_P+d_E+1)
    Theta_ols, _, _, _ = np.linalg.lstsq(Phi_ols, X_all, rcond=None)
    B_init = Theta_ols[:d_P, :].T           # (d_X, d_P)
    C_init = Theta_ols[d_P:d_P + d_E, :].T  # (d_X, d_E)
    d_init = Theta_ols[d_P + d_E, :]        # (d_X,)

    resid_ols = X_all - Phi_ols @ Theta_ols   # (N, d_X)
    n_params_ols = Phi_ols.shape[1]
    Sigma_init = (resid_ols.T @ resid_ols) / max(N - n_params_ols, 1)  # (d_X, d_X)

    # PCA init for A: top d_Z directions of X̃ residuals.
    U, s, Vt = np.linalg.svd(resid_ols, full_matrices=False)
    # Vt rows are right singular vectors; top-d_Z are first d_Z rows.
    A_init = Vt[:dZ, :].T   # (d_X, d_Z)

    # Sigma from full OLS including Z contribution — for now use residual from (P,E) regression.
    # This is approximate; SGD will refine.

    # L_Sigma from Cholesky of Sigma_init (regularized for PSD).
    L_Sigma_raw_init = _cov_to_chol_raw(Sigma_init, d_X)

    # Q, Q_1 init: small diagonal.
    L_Q_raw_init = _small_diag_chol_raw(dZ, cfg.q_init)
    L_Q1_raw_init = _small_diag_chol_raw(dZ, cfg.q_init)

    theta = {
        "beta": np.zeros((dZ, dZ), dtype=float),    # F off-diags: all zero -> diagonal F
        "g_raw": np.zeros((dZ, cfg.n_pi_stim), dtype=float),  # G: zero init
        "b": np.zeros(dZ, dtype=float),
        "m": np.zeros(dZ, dtype=float),
        "A": A_init.copy(),
        "B": B_init.copy(),
        "C": C_init.copy(),
        "d_vec": d_init.copy(),
        "L_Sigma_raw": L_Sigma_raw_init,
        "L_Q_raw": L_Q_raw_init,
        "L_Q1_raw": L_Q1_raw_init,
    }
    return theta


def _cov_to_chol_raw(cov: Array, d: int) -> Array:
    """Convert a PSD covariance to raw Cholesky parameters (inverse of _chol_param_to_cov).

    Uses numpy Cholesky; regularizes if not PSD. Diagonal is passed through
    inverse-softplus to get L_raw diagonal.

    Args:
        cov: (d, d) PSD array.
        d: dimension.

    Returns:
        (d, d) raw lower-triangular array.
    """
    # Regularize: add small ridge until PD.
    eps = 1e-8
    cov_reg = cov.copy() + eps * np.eye(d)
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        L = np.eye(d) * np.sqrt(np.diag(cov_reg).clip(eps))

    # Inverse softplus on diagonal: raw_d such that softplus(raw_d) = L_ii.
    L_raw = L.copy()
    for i in range(d):
        L_raw[i, i] = _inv_softplus(L[i, i])
    return L_raw


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: x such that log(1 + exp(x)) = y.

    Args:
        y: positive value.

    Returns:
        float.
    """
    y = max(y, 1e-8)
    if y > 20.0:
        return y
    return np.log(np.exp(y) - 1.0 + 1e-12)


def _small_diag_chol_raw(d: int, q_init: float) -> Array:
    """Build raw Cholesky params for q_init * I.

    Args:
        d: dimension.
        q_init: diagonal value for Q.

    Returns:
        (d, d) raw lower-triangular.
    """
    L_diag = np.sqrt(q_init)
    L_raw = np.zeros((d, d), dtype=float)
    for i in range(d):
        L_raw[i, i] = _inv_softplus(L_diag)
    return L_raw


def _perturb_theta(theta: dict, seed: int, cfg: JointMLEKalmanConfig) -> dict:
    """Apply small random perturbations to non-deterministic init components.

    Deterministic components (A from PCA, B/C/d from OLS, L_Sigma from OLS
    residuals) are left unchanged. Small-random components: b, m, g_raw,
    and Cholesky off-diagonals.

    Args:
        theta: base theta from _staged_init.
        seed: per-seed integer.
        cfg: config.

    Returns:
        New theta dict with perturbed components.
    """
    rng = np.random.default_rng(seed)
    scale = 0.01
    t = {k: v.copy() for k, v in theta.items()}
    t["b"] = rng.standard_normal(cfg.d_Z) * scale
    t["m"] = rng.standard_normal(cfg.d_Z) * scale
    t["g_raw"] = rng.standard_normal((cfg.d_Z, cfg.n_pi_stim)) * scale
    # Perturb beta (F off-diagonals) slightly.
    t["beta"] = rng.standard_normal((cfg.d_Z, cfg.d_Z)) * scale
    return t


# ---------------------------------------------------------------------------
# Internal: SGD fit
# ---------------------------------------------------------------------------

def _sgd_fit(
    theta: dict,
    mu0_train: dict[str, Array],
    train_ids: list[str],
    val_ids: list[str],
    cohort: Mapping[str, Channels],
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim: Array,
    cfg: JointMLEKalmanConfig,
    seed: int,
    max_rest_days: int = 10,
) -> tuple[dict, dict[str, Array], dict]:
    """SGD optimization of joint NLL over shared theta and per-athlete mu_0.

    Args:
        theta: initial parameter dict (numpy float64).
        mu0_train: initial per-athlete mu_0 (dict of shape (d_Z,) arrays).
        train_ids: subject_ids in training set.
        val_ids: subject_ids in validation set.
        cohort: all athletes' Channels.
        x_names, p_names, e_names: channel names.
        pi_bar_stim: frozen cohort-mean pi_stim, shape (n_pi_stim,).
        cfg: JointMLEKalmanConfig.
        seed: integer seed for torch RNG.
        max_rest_days: consecutive rest days beyond which rows are masked from loss.

    Returns:
        (theta_final, mu0_final, diagnostics_dict) all as numpy.
    """
    import torch
    import torch.optim as optim

    torch.manual_seed(seed)
    dtype = torch.float64

    # Convert theta to torch Parameters.
    t_params = {}
    for k, v in theta.items():
        t_params[k] = torch.nn.Parameter(
            torch.tensor(v, dtype=dtype, requires_grad=True)
        )

    # Per-athlete mu_0 as torch Parameters.
    t_mu0 = {}
    for sid in train_ids:
        t_mu0[sid] = torch.nn.Parameter(
            torch.tensor(mu0_train[sid], dtype=dtype, requires_grad=True)
        )

    pi_bar_t = torch.tensor(pi_bar_stim, dtype=dtype)

    all_params = list(t_params.values()) + list(t_mu0.values())
    optimizer = optim.Adam(all_params, lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_iterations,
        eta_min=cfg.min_learning_rate,
    )

    best_val_ll = -np.inf
    best_state = None
    patience_count = 0
    train_loss_history = []

    for iteration in range(cfg.max_iterations):
        optimizer.zero_grad()

        # Training loss: sum of negative log-likelihoods.
        train_nll = torch.tensor(0.0, dtype=dtype)
        for sid in train_ids:
            ch = cohort[sid]
            nll_i = _athlete_nll_torch(
                t_params=t_params,
                mu0=t_mu0[sid],
                channels=ch,
                x_names=x_names,
                p_names=p_names,
                e_names=e_names,
                pi_bar_stim=pi_bar_t,
                cfg=cfg,
                max_rest_days=max_rest_days,
            )
            if nll_i is not None:
                train_nll = train_nll + nll_i

        train_nll.backward()
        # Gradient clipping for numerical stability.
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        optimizer.step()
        scheduler.step()

        train_loss_history.append(float(train_nll.item()))

        # Validation check.
        if (iteration + 1) % cfg.eval_every == 0 or iteration == cfg.max_iterations - 1:
            with torch.no_grad():
                val_ll = _compute_val_ll(
                    t_params=t_params,
                    t_mu0=t_mu0,
                    val_ids=val_ids,
                    cohort=cohort,
                    x_names=x_names,
                    p_names=p_names,
                    e_names=e_names,
                    pi_bar_stim=pi_bar_t,
                    cfg=cfg,
                    max_rest_days=max_rest_days,
                )

            # Handle initial -inf case: any finite val_ll is an improvement.
            if best_val_ll == -np.inf:
                is_improvement = np.isfinite(val_ll)
            else:
                is_improvement = val_ll > best_val_ll + cfg.tol_conv * abs(best_val_ll + 1e-8)
            if is_improvement:
                best_val_ll = val_ll
                # Save a copy of current parameters.
                best_state = {
                    "theta": {k: v.detach().cpu().numpy().copy() for k, v in t_params.items()},
                    "mu0": {sid: v.detach().cpu().numpy().copy() for sid, v in t_mu0.items()},
                }
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= cfg.patience:
                break

    # Fall back to last state if best_state was never updated.
    if best_state is None:
        best_state = {
            "theta": {k: v.detach().cpu().numpy().copy() for k, v in t_params.items()},
            "mu0": {sid: v.detach().cpu().numpy().copy() for sid, v in t_mu0.items()},
        }
        best_val_ll = float(-np.inf) if best_val_ll == -np.inf else best_val_ll

    diagnostics = {
        "val_log_lik": float(best_val_ll),
        "final_train_nll": float(train_loss_history[-1]) if train_loss_history else float("nan"),
        "n_iterations": iteration + 1,
    }

    return best_state["theta"], best_state["mu0"], diagnostics


# ---------------------------------------------------------------------------
# Internal: per-athlete NLL (torch)
# ---------------------------------------------------------------------------

def _athlete_nll_torch(
    t_params: dict,
    mu0,
    channels: Channels,
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim,
    cfg: JointMLEKalmanConfig,
    max_rest_days: int = 10,
):
    """Compute per-athlete negative marginal log-likelihood via differentiable Kalman filter.

    Args:
        t_params: dict of torch Parameters (shared theta).
        mu0: torch Parameter shape (d_Z,), per-athlete initial mean.
        channels: Channels for this athlete.
        x_names, p_names, e_names: channel names.
        pi_bar_stim: torch tensor shape (n_pi_stim,), frozen.
        cfg: config.
        max_rest_days: consecutive rest days beyond which rows are masked from loss.

    Returns:
        Scalar torch tensor (NLL), or None if no usable rows.
    """
    import torch
    import torch.nn.functional as F_func

    dtype = torch.float64
    dZ = cfg.d_Z
    dX = len(x_names)

    # Build structural parameters from t_params.
    f_diag = torch.tensor(
        [np.exp(-1.0 / t) for t in cfg.tau], dtype=dtype
    )
    h_diag = f_diag.clone()

    beta = t_params["beta"]
    alpha = torch.tanh(beta)
    F_mat = torch.diag(f_diag)
    for i in range(dZ):
        for j in range(dZ):
            if i != j:
                F_mat = F_mat.clone()
                F_mat[i, j] = alpha[i, j] * (1.0 - f_diag[i])

    H_mat = torch.diag(h_diag)

    g_raw = t_params["g_raw"]
    G_mat = g_raw * (1.0 - f_diag).unsqueeze(1)   # (d_Z, n_pi_stim)

    b = t_params["b"]
    m = t_params["m"]
    r_1 = (torch.eye(dZ, dtype=dtype) - H_mat) @ b

    Sigma = _chol_param_to_cov_torch(t_params["L_Sigma_raw"], dX)
    Q = _chol_param_to_cov_torch(t_params["L_Q_raw"], dZ)
    Q_1 = _chol_param_to_cov_torch(t_params["L_Q1_raw"], dZ)

    A = t_params["A"]   # (d_X, d_Z)
    B = t_params["B"]   # (d_X, d_P)
    C = t_params["C"]   # (d_X, d_E)
    d_vec = t_params["d_vec"]  # (d_X,)

    # Diffuse prior covariance.
    Sigma_0 = cfg.sigma0_sq * torch.eye(dZ, dtype=dtype)

    # Initial filter state.
    z_filt = mu0.clone()                # (d_Z,)
    P_filt = Sigma_0.clone()            # (d_Z, d_Z)

    nll = torch.tensor(0.0, dtype=dtype)
    T = channels.dates.shape[0]

    # Count consecutive rest days for rest-bound detection.
    consec_rest = _count_consecutive_rest(channels.X.is_rest)

    # Get pi_stim for workout days.
    pi_stim_all = torch.tensor(
        _get_pi_stim(channels.X.values, x_names),
        dtype=dtype,
    )  # (T, n_pi_stim); NaN on rest rows

    # Get X̃ (transformed observation).
    from statepace.observation import _transform
    x_tilde_all_np = _transform(channels.X.values, x_names)  # (T, d_X)
    x_tilde_all = torch.tensor(x_tilde_all_np, dtype=dtype)

    p_vals = torch.tensor(channels.P.values, dtype=dtype)   # (T, d_P)
    e_vals = torch.tensor(channels.E.values, dtype=dtype)   # (T, d_E)

    for t in range(T):
        is_rest_t = bool(channels.X.is_rest[t])

        if consec_rest[t] > max_rest_days:
            # Mask from loss: skip this row's observation contribution.
            # Re-entry prior is applied at the first post-gap workout day.
            continue

        # Check if this is the first workout day after a rest-bound gap.
        if t > 0 and not is_rest_t and consec_rest[t - 1] > max_rest_days:
            # Re-entry prior: Z ~ N(b, sigma_re^2 * I).
            z_filt = b.clone()
            P_filt = cfg.sigma_re_sq * torch.eye(dZ, dtype=dtype)

        if is_rest_t:
            # Rest-day predict step (no observation update).
            # Z_t = H @ Z_{t-1} + r_1 + noise, noise ~ N(0, Q_1).
            z_pred = H_mat @ z_filt + r_1
            P_pred = H_mat @ P_filt @ H_mat.T + Q_1

            z_filt = z_pred
            P_filt = P_pred
        else:
            # Workout-day predict step.
            pi_stim_t = pi_stim_all[t] - pi_bar_stim   # centered
            z_pred = F_mat @ z_filt + G_mat @ pi_stim_t + m
            P_pred = F_mat @ P_filt @ F_mat.T + Q

            # Observation update.
            x_tilde_t = x_tilde_all[t]   # (d_X,)
            p_t = p_vals[t]               # (d_P,)
            e_t = e_vals[t]               # (d_E,)

            # Predicted observation mean: A @ z_pred + B @ p + C @ e + d.
            mu_x = A @ z_pred + B @ p_t + C @ e_t + d_vec   # (d_X,)

            # Innovation.
            innov = x_tilde_t - mu_x    # (d_X,)

            # Innovation covariance: S = A @ P_pred @ A.T + Sigma.
            S = A @ P_pred @ A.T + Sigma   # (d_X, d_X)

            # Kalman gain: K = P_pred @ A.T @ inv(S).
            try:
                S_chol = torch.linalg.cholesky(S)
                K = torch.cholesky_solve(
                    (P_pred @ A.T).T, S_chol
                ).T   # (d_Z, d_X)
            except Exception:
                # Fallback to pinv if cholesky fails.
                K = P_pred @ A.T @ torch.linalg.pinv(S)

            # NLL contribution: 0.5 * (log|S| + innov.T @ S^{-1} @ innov + d_X * log(2pi)).
            try:
                log_det_S = 2.0 * torch.sum(torch.log(torch.diag(S_chol)))
            except Exception:
                sign, log_det_S = torch.linalg.slogdet(S)
                log_det_S = log_det_S if sign > 0 else torch.tensor(float("inf"), dtype=dtype)

            innov_sq = innov @ torch.linalg.solve(S, innov)
            nll = nll + 0.5 * (log_det_S + innov_sq + dX * np.log(2.0 * np.pi))

            # State update (Joseph form for PSD guarantee).
            z_filt = z_pred + K @ innov
            I_KA = torch.eye(dZ, dtype=dtype) - K @ A
            P_filt = I_KA @ P_pred @ I_KA.T + K @ Sigma @ K.T

    if torch.isnan(nll) or torch.isinf(nll):
        return None
    return nll


def _count_consecutive_rest(is_rest: Array) -> Array:
    """Count consecutive trailing rest days at each timestep.

    Args:
        is_rest: bool array shape (T,).

    Returns:
        Integer array shape (T,); 0 on workout days, run-length on rest days.
    """
    T = len(is_rest)
    count = np.zeros(T, dtype=int)
    for t in range(T):
        if is_rest[t]:
            count[t] = count[t - 1] + 1 if t > 0 else 1
    return count


def _compute_val_ll(
    t_params: dict,
    t_mu0: dict,
    val_ids: list[str],
    cohort: Mapping[str, Channels],
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim,
    cfg: JointMLEKalmanConfig,
    max_rest_days: int = 10,
) -> float:
    """Compute total log-likelihood on validation cohort.

    For validation athletes not in t_mu0 (training-cohort dict), uses mu_0 = 0.

    Args:
        t_params: shared torch Parameters.
        t_mu0: per-athlete mu_0 torch Parameters (training cohort only).
        val_ids: validation subject IDs.
        cohort: all athletes' Channels.
        x_names, p_names, e_names: channel names.
        pi_bar_stim: torch tensor.
        cfg: config.

    Returns:
        Total validation log-likelihood (float).
    """
    import torch
    dtype = torch.float64

    total_ll = 0.0
    for sid in val_ids:
        ch = cohort[sid]
        if sid in t_mu0:
            mu0 = t_mu0[sid]
        else:
            mu0 = torch.zeros(cfg.d_Z, dtype=dtype)
        nll = _athlete_nll_torch(
            t_params=t_params,
            mu0=mu0,
            channels=ch,
            x_names=x_names,
            p_names=p_names,
            e_names=e_names,
            pi_bar_stim=pi_bar_stim,
            cfg=cfg,
            max_rest_days=max_rest_days,
        )
        if nll is not None:
            total_ll -= float(nll.item())

    return total_ll


# ---------------------------------------------------------------------------
# Internal: validation-cohort mu_0 fitting
# ---------------------------------------------------------------------------

def _fit_mu0_single(
    theta: dict,
    channels: Channels,
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim: Array,
    cfg: JointMLEKalmanConfig,
    max_rest_days: int = 10,
) -> Array:
    """Fit mu_0 for a single athlete under frozen theta.

    Same SGD code path as training fit, but only mu_0 is free.

    Args:
        theta: frozen shared parameters (numpy).
        channels: Channels for this athlete.
        x_names, p_names, e_names: channel names.
        pi_bar_stim: frozen cohort-mean pi_stim.
        cfg: config.
        max_rest_days: consecutive rest days beyond which rows are masked.

    Returns:
        Array of shape (d_Z,), fitted mu_0.
    """
    import torch
    import torch.optim as optim

    dtype = torch.float64
    torch.manual_seed(42)

    t_params = {k: torch.tensor(v, dtype=dtype, requires_grad=False) for k, v in theta.items()}
    pi_bar_t = torch.tensor(pi_bar_stim, dtype=dtype)

    mu0 = torch.nn.Parameter(torch.zeros(cfg.d_Z, dtype=dtype))
    optimizer = optim.Adam([mu0], lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.infer_max_iterations, eta_min=cfg.min_learning_rate
    )

    best_nll = np.inf
    best_mu0 = np.zeros(cfg.d_Z)

    for _ in range(cfg.infer_max_iterations):
        optimizer.zero_grad()
        nll = _athlete_nll_torch(
            t_params=t_params,
            mu0=mu0,
            channels=channels,
            x_names=x_names,
            p_names=p_names,
            e_names=e_names,
            pi_bar_stim=pi_bar_t,
            cfg=cfg,
            max_rest_days=max_rest_days,
        )
        if nll is None:
            break
        if float(nll.item()) < best_nll:
            best_nll = float(nll.item())
            best_mu0 = mu0.detach().numpy().copy()
        nll.backward()
        optimizer.step()
        scheduler.step()

    return best_mu0


# ---------------------------------------------------------------------------
# Internal: numpy Kalman filter (for infer after fit)
# ---------------------------------------------------------------------------

def _kalman_filter_numpy(
    theta: dict,
    mu0: Array,
    channels: Channels,
    x_names: tuple[str, ...],
    p_names: tuple[str, ...],
    e_names: tuple[str, ...],
    pi_bar_stim: Array,
    cfg: JointMLEKalmanConfig,
    max_rest_days: int = 10,
) -> tuple[Array, Array]:
    """Run Kalman filter forward pass (numpy, no gradients needed for infer).

    Args:
        theta: fitted parameters (numpy dict).
        mu0: shape (d_Z,), per-athlete initial mean.
        channels: Channels for one athlete.
        x_names, p_names, e_names: channel names.
        pi_bar_stim: shape (n_pi_stim,), frozen.
        cfg: config.
        max_rest_days: consecutive rest days beyond which rows are masked.

    Returns:
        (mean_traj, cov_traj): shapes (T, d_Z) and (T, d_Z, d_Z).
    """
    arrays = _theta_to_arrays(theta, cfg)
    F = arrays["F"]
    H = arrays["H"]
    G = arrays["G"]
    b = arrays["b"]
    r_1 = arrays["r_1"]
    m = arrays["m"]
    Sigma = arrays["Sigma"]
    Q = arrays["Q"]
    Q_1 = arrays["Q_1"]
    A = arrays["A"]
    B = arrays["B"]
    C = arrays["C"]
    d_vec = arrays["d_vec"]

    dZ = cfg.d_Z
    dX = len(x_names)
    T = channels.dates.shape[0]

    Sigma_0 = cfg.sigma0_sq * np.eye(dZ)

    z_filt = mu0.copy()
    P_filt = Sigma_0.copy()

    mean_traj = np.zeros((T, dZ))
    cov_traj = np.zeros((T, dZ, dZ))

    from statepace.observation import _transform
    x_tilde_all = _transform(channels.X.values, x_names)
    pi_stim_all = _get_pi_stim(channels.X.values, x_names)
    p_vals = channels.P.values
    e_vals = channels.E.values

    consec_rest = _count_consecutive_rest(channels.X.is_rest)

    for t in range(T):
        is_rest_t = bool(channels.X.is_rest[t])

        if consec_rest[t] > max_rest_days:
            mean_traj[t] = z_filt
            cov_traj[t] = P_filt
            continue

        if t > 0 and not is_rest_t and consec_rest[t - 1] > max_rest_days:
            z_filt = b.copy()
            P_filt = cfg.sigma_re_sq * np.eye(dZ)

        if is_rest_t:
            z_pred = H @ z_filt + r_1
            P_pred = H @ P_filt @ H.T + Q_1
            z_filt = z_pred
            P_filt = P_pred
        else:
            pi_stim_t = pi_stim_all[t] - pi_bar_stim
            z_pred = F @ z_filt + G @ pi_stim_t + m
            P_pred = F @ P_filt @ F.T + Q

            x_tilde_t = x_tilde_all[t]
            p_t = p_vals[t]
            e_t = e_vals[t]

            mu_x = A @ z_pred + B @ p_t + C @ e_t + d_vec
            innov = x_tilde_t - mu_x

            S = A @ P_pred @ A.T + Sigma
            try:
                S_chol = np.linalg.cholesky(S + 1e-10 * np.eye(dX))
                K = np.linalg.solve(S_chol.T, np.linalg.solve(S_chol, (P_pred @ A.T).T)).T
            except np.linalg.LinAlgError:
                K = P_pred @ A.T @ np.linalg.pinv(S)

            z_filt = z_pred + K @ innov
            I_KA = np.eye(dZ) - K @ A
            P_filt = I_KA @ P_pred @ I_KA.T + K @ Sigma @ K.T

        mean_traj[t] = z_filt
        cov_traj[t] = P_filt

    return mean_traj, cov_traj


# ---------------------------------------------------------------------------
# Internal: post-hoc sign cascade
# ---------------------------------------------------------------------------

def _apply_sign_cascade(theta: dict, cfg: JointMLEKalmanConfig) -> dict:
    """Apply post-hoc sign cascade per spec §Post-hoc gauge convention.

    For each dimension i:
      1. If |b_i| >= tol_b: flip if b_i < 0.
      2. Elif |m_i| >= tol_m: flip if m_i < 0.
      3. Elif sum(G[i, :]) < 0: flip.
      4. Else: no flip (dimension is at gauge boundary).

    Flip of dimension i: negate A[:, i], b[i], m[i], G[i, :],
    L_Q_raw[i, :] off-diagonal rows, L_Q1_raw[i, :] off-diagonal rows.
    beta[i, :] (alpha -> -alpha_ij for row i, but applied to off-diagonals only).

    This is applied post-fit for diagnostics only.

    Args:
        theta: parameter dict (numpy).
        cfg: config.

    Returns:
        New theta dict with sign convention applied.
    """
    t = {k: v.copy() for k, v in theta.items()}
    dZ = cfg.d_Z

    b = t["b"]
    m_vec = t["m"]
    g_raw = t["g_raw"]   # (d_Z, n_pi_stim); G_ij = (1-F_ii)*g_ij -> sign(G[i,:]) = sign(g_raw[i,:])

    for i in range(dZ):
        if abs(b[i]) >= cfg.tol_b:
            do_flip = b[i] < 0
        elif abs(m_vec[i]) >= cfg.tol_m:
            do_flip = m_vec[i] < 0
        else:
            do_flip = float(np.sum(g_raw[i, :])) < 0

        if do_flip:
            t["b"][i] *= -1.0
            t["m"][i] *= -1.0
            t["g_raw"][i, :] *= -1.0
            t["A"][:, i] *= -1.0
            # L_Q_raw: flip row i of lower triangle (off-diagonals; diagonal is via softplus so positive).
            # Flipping L_Q_raw[i, j] for j < i and L_Q_raw[j, i] for j > i.
            t["L_Q_raw"][i, :i] *= -1.0
            t["L_Q_raw"][i + 1:, i] *= -1.0
            t["L_Q1_raw"][i, :i] *= -1.0
            t["L_Q1_raw"][i + 1:, i] *= -1.0
            # beta row i (F off-diagonals): flip sign means negating alpha -> negate beta row.
            t["beta"][i, :i] *= -1.0
            t["beta"][i, i + 1:] *= -1.0

    return t


def _find_gauge_boundary_dims(theta: dict, cfg: JointMLEKalmanConfig) -> list[int]:
    """Find dimensions at the gauge boundary (all three cascade conditions below tolerance).

    Args:
        theta: parameter dict after sign cascade.
        cfg: config.

    Returns:
        List of dimension indices at the gauge boundary.
    """
    b = theta["b"]
    m_vec = theta["m"]
    g_raw = theta["g_raw"]
    boundary = []
    for i in range(cfg.d_Z):
        if (abs(b[i]) < cfg.tol_b
                and abs(m_vec[i]) < cfg.tol_m
                and abs(float(np.sum(g_raw[i, :]))) < cfg.tol_b):
            boundary.append(i)
    return boundary
