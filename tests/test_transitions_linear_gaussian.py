"""Tests for LinearGaussianWorkoutTransition and LinearGaussianRestTransition.

Spec: docs/reference_impls/linear-gaussian.md
Fixture: make_linear_gaussian_cohort (tests/fixtures/reference_impls/linear_gaussian.py)
All tests use real Channels + synthetic Z; no mocking.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.reference_impls.linear_gaussian import make_linear_gaussian_cohort
from tests.fixtures.reference_impls.riegel_score_hrstep import make_riegel_hrstep_cohort
from statepace.channels import X as X_ch, Z
from statepace.transitions import (
    LinearGaussianWorkoutTransition,
    LinearGaussianRestTransition,
)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_N_ATHLETES = 8
_N_DAYS = 150
_D_Z = 3
_MAX_REST = 7
_SEED = 77


# ---------------------------------------------------------------------------
# Cohort helpers
# ---------------------------------------------------------------------------

def _make_cohort():
    return make_linear_gaussian_cohort(_N_ATHLETES, _N_DAYS, _D_Z, seed=_SEED)


def _concat_channels_and_z(cohort, z_traj):
    """Concatenate all athletes' Channels and Z trajectories into single arrays.

    Builds Z_prev (rows 0..T-2 per athlete), Z_t (rows 1..T-1), and the
    corresponding X_t slice, all concatenated across athletes.

    Returns:
        (Z_prev_cat, X_t_cat, Z_t_cat): Z and X_ch dataclasses with rows
        concatenated across athletes, aligned as (t-1, t) pairs.
    """
    z_prev_parts, x_t_parts, z_t_parts = [], [], []
    is_rest_parts = []

    for sid, ch in cohort.items():
        z = z_traj[sid]
        T = _N_DAYS

        # Pairs: (t-1, t) for t in 1..T-1
        z_prev_parts.append(z.mean[:-1])               # (T-1, d_Z)
        z_t_parts.append(z.mean[1:])                   # (T-1, d_Z)
        x_t_parts.append(ch.X.values[1:])              # (T-1, d_X)
        is_rest_parts.append(ch.X.is_rest[1:])         # (T-1,)

    z_prev_mean = np.concatenate(z_prev_parts, axis=0)
    z_t_mean = np.concatenate(z_t_parts, axis=0)
    x_vals = np.concatenate(x_t_parts, axis=0)
    is_rest = np.concatenate(is_rest_parts, axis=0)

    T_cat = z_prev_mean.shape[0]
    dummy_dates = np.zeros(T_cat, dtype="datetime64[ns]")

    sample_ch = next(iter(cohort.values()))
    x_cat = X_ch(values=x_vals, names=sample_ch.X.names, is_rest=is_rest)
    z_prev_cat = Z(mean=z_prev_mean, cov=None, dates=dummy_dates)
    z_t_cat = Z(mean=z_t_mean, cov=None, dates=dummy_dates)
    return z_prev_cat, x_cat, z_t_cat


def _compute_n_rest_days(is_rest: np.ndarray) -> np.ndarray:
    """Compute per-row consecutive trailing rest-day count.

    The i-th element is the run-length of consecutive True values in
    is_rest ending at position i (inclusive). Workout days (is_rest=False)
    get 0.

    Args:
        is_rest: bool array of shape (T,).

    Returns:
        Integer array of shape (T,).
    """
    T = len(is_rest)
    n = np.zeros(T, dtype=int)
    for i in range(T):
        if is_rest[i]:
            n[i] = n[i - 1] + 1 if i > 0 else 1
    return n


def _make_rest_inputs(cohort, z_traj):
    """Build concatenated (Z_prev, n_rest_days, Z_t) for the rest transition tests.

    Includes only rest-day rows (n_rest_days >= 1) from all athletes.

    Returns:
        (Z_prev_cat, n_rest_days_cat, Z_t_cat): arrays aligned across athletes.
    """
    z_prev_parts, n_rest_parts, z_t_parts = [], [], []

    for sid, ch in cohort.items():
        z = z_traj[sid]
        T = _N_DAYS

        n_rest = _compute_n_rest_days(ch.X.is_rest)   # (T,)

        # Pairs (t-1, t): only rest-day rows
        for t in range(1, T):
            if ch.X.is_rest[t]:
                z_prev_parts.append(z.mean[t - 1])
                n_rest_parts.append(n_rest[t])
                z_t_parts.append(z.mean[t])

    z_prev_mean = np.array(z_prev_parts, dtype=float)      # (N, d_Z)
    n_rest_arr = np.array(n_rest_parts, dtype=int)         # (N,)
    z_t_mean = np.array(z_t_parts, dtype=float)            # (N, d_Z)

    N = z_prev_mean.shape[0]
    dummy_dates = np.zeros(N, dtype="datetime64[ns]")
    z_prev_cat = Z(mean=z_prev_mean, cov=None, dates=dummy_dates)
    z_t_cat = Z(mean=z_t_mean, cov=None, dates=dummy_dates)
    return z_prev_cat, n_rest_arr, z_t_cat


# ---------------------------------------------------------------------------
# Workout transition tests
# ---------------------------------------------------------------------------

def test_workout_fit_shapes():
    """fit() produces F(d_Z, d_Z), G(d_Z, 5), m(d_Z,), Q(d_Z, d_Z) symmetric PSD."""
    cohort, z_traj = _make_cohort()
    z_prev, x_t, z_t = _concat_channels_and_z(cohort, z_traj)

    model = LinearGaussianWorkoutTransition(d_Z=_D_Z).fit(z_prev, x_t, z_t)

    assert model.F.shape == (_D_Z, _D_Z), f"F shape {model.F.shape}"
    assert model.G.shape == (_D_Z, 5), f"G shape {model.G.shape}"
    assert model.m.shape == (_D_Z,), f"m shape {model.m.shape}"
    assert model.Q.shape == (_D_Z, _D_Z), f"Q shape {model.Q.shape}"

    np.testing.assert_allclose(model.Q, model.Q.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(model.Q)
    assert np.all(eigvals >= -1e-10), f"Q not PSD; min eigval = {eigvals.min():.3e}"


def test_workout_step_shape():
    """step() returns Z.mean with shape (T, d_Z)."""
    cohort, z_traj = _make_cohort()
    z_prev, x_t, z_t = _concat_channels_and_z(cohort, z_traj)

    model = LinearGaussianWorkoutTransition(d_Z=_D_Z).fit(z_prev, x_t, z_t)
    z_out = model.step(z_prev, x_t)

    T = z_prev.mean.shape[0]
    assert z_out.mean.shape == (T, _D_Z), f"step mean shape {z_out.mean.shape}"
    assert z_out.cov is None


def test_workout_log_prob_shape_and_rest_zero():
    """log_prob() returns (T,); rest rows exactly 0.0; non-rest rows finite."""
    cohort, z_traj = _make_cohort()
    z_prev, x_t, z_t = _concat_channels_and_z(cohort, z_traj)

    model = LinearGaussianWorkoutTransition(d_Z=_D_Z).fit(z_prev, x_t, z_t)
    lp = model.log_prob(z_prev, x_t, z_t)

    T = z_t.mean.shape[0]
    assert lp.shape == (T,), f"log_prob shape {lp.shape}"

    rest_mask = x_t.is_rest
    if np.any(rest_mask):
        np.testing.assert_array_equal(
            lp[rest_mask], 0.0,
            err_msg="log_prob on rest rows must be exactly 0.0",
        )

    non_rest_lp = lp[~rest_mask]
    assert np.all(np.isfinite(non_rest_lp)), (
        f"Non-rest log_prob has non-finite values: {non_rest_lp[~np.isfinite(non_rest_lp)]}"
    )


def test_workout_fit_deterministic():
    """fit() twice on identical input produces element-wise identical parameters."""
    cohort, z_traj = _make_cohort()
    z_prev, x_t, z_t = _concat_channels_and_z(cohort, z_traj)

    model1 = LinearGaussianWorkoutTransition(d_Z=_D_Z).fit(z_prev, x_t, z_t)
    model2 = LinearGaussianWorkoutTransition(d_Z=_D_Z).fit(z_prev, x_t, z_t)

    np.testing.assert_array_equal(model1.F, model2.F)
    np.testing.assert_array_equal(model1.G, model2.G)
    np.testing.assert_array_equal(model1.m, model2.m)
    np.testing.assert_array_equal(model1.Q, model2.Q)


# ---------------------------------------------------------------------------
# Rest transition tests
# ---------------------------------------------------------------------------

def test_rest_fit_shapes_and_constraints():
    """fit() produces H(d_Z,d_Z) symmetric with eigenvalues in (0,1); r_1(d_Z,); Q_1 symmetric PSD."""
    cohort, z_traj = _make_cohort()
    z_prev_r, n_rest, z_t_r = _make_rest_inputs(cohort, z_traj)

    model = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=_MAX_REST).fit(
        z_prev_r, n_rest, z_t_r
    )

    assert model.H.shape == (_D_Z, _D_Z), f"H shape {model.H.shape}"
    assert model.r_1.shape == (_D_Z,), f"r_1 shape {model.r_1.shape}"
    assert model.Q_1.shape == (_D_Z, _D_Z), f"Q_1 shape {model.Q_1.shape}"

    # H symmetric
    h_asym = np.linalg.norm(model.H - model.H.T)
    assert h_asym < 1e-10, f"H not symmetric; ||H - H.T|| = {h_asym:.3e}"

    # H eigenvalues strictly in (0, 1)
    h_eigs = np.linalg.eigvalsh(model.H)
    assert np.all(h_eigs > 0), f"H has eigenvalue <= 0: {h_eigs}"
    assert np.all(h_eigs < 1), f"H has eigenvalue >= 1: {h_eigs}"

    # Q_1 symmetric PSD
    np.testing.assert_allclose(model.Q_1, model.Q_1.T, atol=1e-10)
    q1_eigs = np.linalg.eigvalsh(model.Q_1)
    assert np.all(q1_eigs >= -1e-10), f"Q_1 not PSD; min eigval = {q1_eigs.min():.3e}"


def test_rest_step_at_n1_matches_linear_model():
    """step(Z_prev, 1).mean ≈ Z_prev.mean @ H.T + r_1 (within float tolerance)."""
    cohort, z_traj = _make_cohort()
    z_prev_r, n_rest, z_t_r = _make_rest_inputs(cohort, z_traj)

    model = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=_MAX_REST).fit(
        z_prev_r, n_rest, z_t_r
    )

    # Use a small subset for speed; pick first 20 rows.
    sub = Z(mean=z_prev_r.mean[:20], cov=None, dates=z_prev_r.dates[:20])
    z_step = model.step(sub, 1)

    expected = sub.mean @ model.H.T + model.r_1
    np.testing.assert_allclose(z_step.mean, expected, atol=1e-12)


def test_rest_step_shape_and_bound():
    """step() returns correct shape; raises on n > max or n < 1."""
    cohort, z_traj = _make_cohort()
    z_prev_r, n_rest, z_t_r = _make_rest_inputs(cohort, z_traj)

    model = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=_MAX_REST).fit(
        z_prev_r, n_rest, z_t_r
    )

    sub = Z(mean=z_prev_r.mean[:10], cov=None, dates=z_prev_r.dates[:10])

    # Valid n in range.
    for n in range(1, _MAX_REST + 1):
        z_out = model.step(sub, n)
        assert z_out.mean.shape == (10, _D_Z), f"shape at n={n}: {z_out.mean.shape}"

    # n too large.
    with pytest.raises(ValueError):
        model.step(sub, _MAX_REST + 1)

    # n < 1.
    with pytest.raises(ValueError):
        model.step(sub, 0)


def test_rest_log_prob_shape():
    """log_prob() returns (T,) finite values for n in valid range."""
    cohort, z_traj = _make_cohort()
    z_prev_r, n_rest, z_t_r = _make_rest_inputs(cohort, z_traj)

    model = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=_MAX_REST).fit(
        z_prev_r, n_rest, z_t_r
    )

    sub_prev = Z(mean=z_prev_r.mean[:30], cov=None, dates=z_prev_r.dates[:30])
    sub_t = Z(mean=z_t_r.mean[:30], cov=None, dates=z_t_r.dates[:30])

    lp = model.log_prob(sub_prev, 1, sub_t)
    assert lp.shape == (30,), f"log_prob shape {lp.shape}"
    assert np.all(np.isfinite(lp)), f"log_prob has non-finite values: {lp[~np.isfinite(lp)]}"
