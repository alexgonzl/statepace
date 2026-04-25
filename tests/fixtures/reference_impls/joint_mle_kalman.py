"""Fixture factory for the joint-mle-kalman reference impl spec.

Serves: docs/reference_impls/joint-mle-kalman.md
Used by: tests/test_filter_joint_mle_kalman.py

Generates synthetic cohort data FROM the M6 SSM model (data-generating process
matches the structural parameterization). This is required for the NA1 validation
mu_0 recovery test.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from statepace.channels import Channels, Z
from statepace.channels import E as E_channel
from statepace.channels import P as P_channel
from statepace.channels import X as X_channel

# Channel schemas must match the paired riegel-score-hrstep impl.
_P_NAMES: tuple[str, ...] = (
    "total_distance",
    "total_duration",
    "elevation",
    "is_track",
    "time_of_day_sin",
    "time_of_day_cos",
)
_E_NAMES: tuple[str, ...] = ("wet_bulb_temp",)
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
_PI_STIM_NAMES: tuple[str, ...] = (
    "hr_load",
    "step_load",
    "total_elevation_gain",
    "total_elevation_lost",
    "heat_exposure",
)

_REST_FRAC = 0.20


def _build_ground_truth_theta(
    d_Z: int,
    tau: tuple[float, ...],
    n_pi_stim: int,
    rng: np.random.Generator,
) -> dict:
    """Build a plausible ground-truth theta for the M6 SSM.

    Structural constraints:
      H fully diagonal: H_ii = exp(-1/tau_i)
      F diagonal fixed: F_ii = exp(-1/tau_i); off-diags = 0.1 * (1 - F_ii)
      G_ij = (1 - F_ii) * g_ij, g_ij small random
      r_1 = (I - H) @ b
      Sigma, Q, Q_1 small PD

    Args:
        d_Z: latent dimensionality.
        tau: time constants.
        n_pi_stim: number of stimulus channels.
        rng: seeded RNG.

    Returns:
        Dict with keys: F, H, G, b, r_1, m, Sigma, Q, Q_1, A, B, C, d_vec.
    """
    tau_arr = np.array(tau, dtype=float)
    f_diag = np.exp(-1.0 / tau_arr)
    h_diag = f_diag.copy()

    # F: diagonal + small off-diagonals.
    F = np.diag(f_diag)
    for i in range(d_Z):
        for j in range(d_Z):
            if i != j:
                F[i, j] = 0.05 * (1.0 - f_diag[i]) * rng.standard_normal()

    H = np.diag(h_diag)

    # b: detraining target, small positive.
    b = 0.2 + 0.1 * np.abs(rng.standard_normal(d_Z))

    # r_1 derived.
    r_1 = (np.eye(d_Z) - H) @ b

    # m: workout drift, small.
    m = 0.05 * rng.standard_normal(d_Z)

    # G: stimulus loading.
    g_raw = 0.1 * rng.standard_normal((d_Z, n_pi_stim))
    G = g_raw * (1.0 - f_diag)[:, np.newaxis]

    # Covariances: very small PD to keep Z trajectories bounded.
    def small_pd(d, scale):
        A_mat = rng.standard_normal((d, d)) * scale
        return A_mat @ A_mat.T + scale * np.eye(d)

    Sigma = small_pd(10, 0.005)
    Q = small_pd(d_Z, 0.001)
    Q_1 = small_pd(d_Z, 0.0005)

    # Observation parameters: A must dominate B and C in transformed observation space so
    # the latent state is identifiable. With Z trajectories O(1), A~1.0 gives state
    # contribution ~1.0. B~1e-5 keeps planned-condition contribution negligible even
    # when p_vals are O(1e3-1e4). C~1e-3 similarly small for exogenous channels.
    A = 1.0 * rng.standard_normal((10, d_Z))
    B = 1e-5 * rng.standard_normal((10, len(_P_NAMES)))
    C = 1e-3 * rng.standard_normal((10, 1))
    d_vec = rng.standard_normal(10) * 0.1

    return dict(F=F, H=H, G=G, b=b, r_1=r_1, m=m,
                Sigma=Sigma, Q=Q, Q_1=Q_1, A=A, B=B, C=C, d_vec=d_vec)


def _generate_trajectory(
    theta: dict,
    mu0: np.ndarray,
    sigma0_sq: float,
    is_rest: np.ndarray,
    p_vals: np.ndarray,
    e_vals: np.ndarray,
    pi_stim_vals: np.ndarray,
    x_names: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Z and X̃ trajectories from the M6 SSM.

    Args:
        theta: ground-truth parameters.
        mu0: shape (d_Z,), true initial state.
        sigma0_sq: initial covariance scale.
        is_rest: bool array shape (T,).
        p_vals: shape (T, d_P).
        e_vals: shape (T, d_E).
        pi_stim_vals: shape (T, n_pi_stim).
        x_names: X component names.
        rng: seeded RNG.

    Returns:
        (z_traj, x_tilde_traj): shapes (T, d_Z) and (T, d_X).
    """
    F = theta["F"]
    H = theta["H"]
    G = theta["G"]
    b = theta["b"]
    r_1 = theta["r_1"]
    m = theta["m"]
    Sigma = theta["Sigma"]
    Q = theta["Q"]
    Q_1 = theta["Q_1"]
    A = theta["A"]
    B = theta["B"]
    C_mat = theta["C"]
    d_vec = theta["d_vec"]

    d_Z = F.shape[0]
    d_X = A.shape[0]
    T = is_rest.shape[0]

    # Sample initial Z.
    z_t = rng.multivariate_normal(
        mean=mu0, cov=sigma0_sq * np.eye(d_Z)
    )

    z_traj = np.zeros((T, d_Z))
    x_tilde_traj = np.zeros((T, d_X))

    # pi_bar_stim: we use zero here (generator doesn't need centering).
    pi_bar = np.zeros(G.shape[1])

    for t in range(T):
        # Generate X̃ first (observation depends on z_{t-1} implicitly via z_t here
        # but we generate in SSM order: predict z_t from z_{t-1}, then observe).
        if not is_rest[t]:
            pi_stim_t = pi_stim_vals[t] - pi_bar
            z_t = F @ z_t + G @ pi_stim_t + m + rng.multivariate_normal(
                mean=np.zeros(d_Z), cov=Q
            )
        else:
            z_t = H @ z_t + r_1 + rng.multivariate_normal(
                mean=np.zeros(d_Z), cov=Q_1
            )

        z_traj[t] = z_t

        if not is_rest[t]:
            mu_x = A @ z_t + B @ p_vals[t] + C_mat @ e_vals[t] + d_vec
            x_tilde = mu_x + rng.multivariate_normal(
                mean=np.zeros(d_X), cov=Sigma
            )
            x_tilde_traj[t] = x_tilde
        else:
            x_tilde_traj[t] = np.nan

    return z_traj, x_tilde_traj


def _inverse_transform_x(x_tilde: np.ndarray, x_names: tuple[str, ...]) -> np.ndarray:
    """Invert X̃ transforms to get raw X values.

    Needed so the Channels.X values are in the expected domain (positive for
    log-transformed channels). Clips transformed values to a safe range before
    exponentiating to prevent overflow from large SSM-generated values.

    Args:
        x_tilde: shape (T, d_X), transformed values (NaN on rest rows).
        x_names: component names.

    Returns:
        shape (T, d_X), raw-space values (NaN on rest rows).
    """
    x_raw = x_tilde.copy()
    n = {name: i for i, name in enumerate(x_names)}

    def safe_apply(col, fn, clip_lo=None, clip_hi=None):
        mask = ~np.isnan(col)
        out = col.copy()
        vals = col[mask]
        if clip_lo is not None:
            vals = np.clip(vals, clip_lo, clip_hi)
        out[mask] = fn(vals)
        return out

    # Clip log-space to [-10, 10] before exp to prevent overflow.
    x_raw[:, n["best_effort_riegel_speed_score"]] = safe_apply(
        x_tilde[:, n["best_effort_riegel_speed_score"]], np.exp, clip_lo=-10.0, clip_hi=10.0
    )
    x_raw[:, n["hr_load"]] = safe_apply(
        x_tilde[:, n["hr_load"]], lambda v: v * 7200.0
    )
    x_raw[:, n["step_load"]] = safe_apply(
        x_tilde[:, n["step_load"]], lambda v: v * 16200.0
    )
    # log1p-space capped at 20 before expm1 to avoid overflow.
    x_raw[:, n["total_elevation_gain"]] = safe_apply(
        x_tilde[:, n["total_elevation_gain"]], np.expm1, clip_lo=-1.0, clip_hi=20.0
    )
    x_raw[:, n["total_elevation_lost"]] = safe_apply(
        x_tilde[:, n["total_elevation_lost"]], np.expm1, clip_lo=-1.0, clip_hi=20.0
    )
    x_raw[:, n["heat_exposure"]] = safe_apply(
        x_tilde[:, n["heat_exposure"]], np.expm1, clip_lo=-1.0, clip_hi=20.0
    )

    # Clip to physical bounds to avoid negatives from small-sigma noise.
    x_raw[:, n["best_effort_riegel_speed_score"]] = np.where(
        np.isnan(x_raw[:, n["best_effort_riegel_speed_score"]]),
        np.nan,
        np.clip(x_raw[:, n["best_effort_riegel_speed_score"]], 1e-6, None),
    )
    for key in ("hr_load", "step_load", "total_elevation_gain",
                "total_elevation_lost", "heat_exposure"):
        col = x_raw[:, n[key]]
        x_raw[:, n[key]] = np.where(np.isnan(col), np.nan, np.clip(col, 0.0, None))

    return x_raw


def make_joint_mle_kalman_cohort(
    n_athletes: int,
    n_days: int,
    d_Z: int,
    tau: tuple[float, ...],
    sigma0_sq: float,
    seed: int,
) -> tuple[Mapping[str, Channels], Mapping[str, np.ndarray], dict]:
    """Synthetic cohort generated FROM the M6 SSM (for NA1 recovery test).

    Generates Channels, per-athlete true mu_0^{(i)}, and ground-truth theta.
    The Z trajectories are generated from the SSM using the ground-truth theta;
    X observations are generated from the observation model. All channel names
    match the riegel-score-hrstep / linear-gaussian pairing.

    Args:
        n_athletes: number of athletes.
        n_days: number of days (rows) per athlete.
        d_Z: latent dimensionality.
        tau: time constants tuple, length d_Z.
        sigma0_sq: initial state covariance scale (determines spread of mu_0 draws).
        seed: integer seed for np.random.default_rng.

    Returns:
        (cohort, true_mu0, ground_truth_theta) where:
          cohort: Mapping[subject_id, Channels] with SSM-generated observations.
          true_mu0: Mapping[subject_id, np.ndarray] shape (d_Z,) per athlete.
          ground_truth_theta: dict of ground-truth SSM parameters.
    """
    rng = np.random.default_rng(seed)
    n_pi_stim = len(_PI_STIM_NAMES)

    theta = _build_ground_truth_theta(d_Z, tau, n_pi_stim, rng)

    start = np.datetime64("2023-01-01", "D")
    dates = np.arange(start, start + n_days, dtype="datetime64[D]").astype("datetime64[ns]")

    cohort: dict[str, Channels] = {}
    true_mu0: dict[str, np.ndarray] = {}

    for i in range(n_athletes):
        sid = f"A_{i:02d}"

        # Per-athlete true mu_0.
        mu0_i = rng.multivariate_normal(
            mean=np.zeros(d_Z),
            cov=sigma0_sq * np.eye(d_Z),
        )
        true_mu0[sid] = mu0_i

        # Rest schedule.
        is_rest = rng.random(n_days) < _REST_FRAC

        # Sample P.
        total_distance = rng.uniform(2_000, 20_000, size=n_days)
        total_duration = rng.uniform(600, 7_200, size=n_days)
        elevation = rng.uniform(0, 2_500, size=n_days)
        is_track = rng.integers(0, 2, size=n_days).astype(float)
        tod_angle = rng.uniform(0, 2 * np.pi, size=n_days)
        p_vals = np.column_stack([
            total_distance, total_duration, elevation, is_track,
            np.sin(tod_angle), np.cos(tod_angle),
        ])  # (T, 6)

        # Sample E.
        wet_bulb = rng.uniform(0, 30, size=(n_days, 1))
        e_vals = wet_bulb  # (T, 1)

        # Build pi_stim values (raw, before centering — centering applied at fit).
        hr_load = rng.uniform(1_500, 22_000, size=n_days)
        step_load = rng.uniform(3_000, 50_000, size=n_days)
        elev_gain = rng.uniform(0, 1_000, size=n_days)
        elev_lost = rng.uniform(0, 1_000, size=n_days)
        duration_hours = total_duration / 3_600.0
        excess_temp = np.maximum(0.0, wet_bulb[:, 0] - 18.0)
        heat_exposure = np.maximum(0.0, excess_temp * duration_hours * rng.uniform(0.8, 1.2, size=n_days))

        # pi_stim values (T, 5) — transform applied inside _generate_trajectory as raw values.
        # The generator uses raw pi_stim values (no pre-transform needed for load channels in SSM).
        pi_stim_vals = np.column_stack([hr_load, step_load, elev_gain, elev_lost, heat_exposure])  # (T, 5)

        # For X̃ generation, we need the pre-transformed pi_stim. The log1p/scale
        # transforms in _transform operate on X values. Since pi_stim channels are
        # also in X, apply the same transforms used in observation._transform.
        # hr_load -> hr_load/7200; step_load -> step_load/16200;
        # elev_gain, elev_lost, heat_exposure -> log1p.
        pi_stim_transformed = np.column_stack([
            hr_load / 7200.0,
            step_load / 16200.0,
            np.log1p(elev_gain),
            np.log1p(elev_lost),
            np.log1p(heat_exposure),
        ])  # (T, 5)

        # Generate Z and X̃ from SSM.
        z_traj, x_tilde_traj = _generate_trajectory(
            theta=theta,
            mu0=mu0_i,
            sigma0_sq=sigma0_sq,
            is_rest=is_rest,
            p_vals=p_vals,
            e_vals=e_vals,
            pi_stim_vals=pi_stim_transformed,
            x_names=_X_NAMES,
            rng=rng,
        )

        # Build full X̃ array (T, 10): riegel_score_log, grade, mean_hr, hr_drift, speed_cad, hr_load~, step~, elev+, elev-, heat~.
        # x_tilde from generator is (T, 10). We need to build a raw X array
        # by inverting the pre-transforms, then package as Channels.X.
        score_log = rng.lognormal(mean=0.0, sigma=0.15, size=n_days)  # additional variation
        grade = rng.uniform(-0.10, 0.10, size=n_days)
        mean_hr = rng.uniform(120, 180, size=n_days)
        hr_drift = rng.uniform(-15, 15, size=n_days)
        speed_cad = rng.uniform(0.8, 1.6, size=n_days)

        # Construct x_tilde from generated SSM observation + passthrough channels.
        # SSM generates all 10 d_X channels jointly. The passthrough channels
        # (grade, mean_hr, hr_drift, speed_cad) are generated from the
        # multivariate Gaussian in the SSM. For simplicity in the fixture,
        # we use the SSM-generated x_tilde directly (all 10 channels from theta["A"] etc.).
        # x_tilde_traj has shape (T, 10) from _generate_trajectory.

        x_raw = _inverse_transform_x(x_tilde_traj.copy(), _X_NAMES)
        x_raw[is_rest, :] = np.nan

        cohort[sid] = Channels(
            subject_id=sid,
            dates=dates.copy(),
            P=P_channel(values=p_vals, names=_P_NAMES),
            X=X_channel(values=x_raw, names=_X_NAMES, is_rest=is_rest),
            E=E_channel(values=e_vals, names=_E_NAMES),
        )

    return cohort, true_mu0, theta
