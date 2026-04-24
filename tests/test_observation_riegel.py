"""Tests for RiegelScoreHRStep ObservationModel (docs/reference_impls/riegel-score-hrstep.md).

Fixture: make_riegel_hrstep_cohort from tests/fixtures/reference_impls/riegel_score_hrstep.py.
All tests use real Channels; no mocking.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.reference_impls.riegel_score_hrstep import make_riegel_hrstep_cohort
from statepace.channels import Z
from statepace.observation import RiegelScoreHRStep, ConditioningSpec

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_N_ATHLETES = 10
_N_DAYS = 120
_SEED = 42

# Expected X component names per spec.
_X_NAMES = (
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


def _make_cohort():
    return make_riegel_hrstep_cohort(_N_ATHLETES, _N_DAYS, seed=_SEED)


def _concat_channels(cohort):
    """Concatenate all athletes' Channels into single (P, X, E) arrays.

    Returns:
        (Z_prev, P_cat, X_cat, E_cat): Channels objects with rows
        concatenated across all athletes.
    """
    from statepace.channels import P as P_ch, X as X_ch, E as E_ch

    p_vals, x_vals, e_vals, is_rest_all = [], [], [], []
    for ch in cohort.values():
        p_vals.append(ch.P.values)
        x_vals.append(ch.X.values)
        e_vals.append(ch.E.values)
        is_rest_all.append(ch.X.is_rest)

    p_cat = np.concatenate(p_vals, axis=0)
    x_cat = np.concatenate(x_vals, axis=0)
    e_cat = np.concatenate(e_vals, axis=0)
    is_rest_cat = np.concatenate(is_rest_all, axis=0)

    T = p_cat.shape[0]
    # Z_prev: synthetic N(0,I) — no estimator at M4; testing OLS mechanics only.
    # This choice is documented here to make the test's purpose explicit.
    rng = np.random.default_rng(_SEED + 1)
    z_mean = rng.standard_normal((T, 4))
    z_prev = Z(mean=z_mean, cov=None, dates=np.zeros(T, dtype="datetime64[ns]"))

    sample = next(iter(cohort.values()))
    p_out = P_ch(values=p_cat, names=sample.P.names)
    x_out = X_ch(values=x_cat, names=sample.X.names, is_rest=is_rest_cat)
    e_out = E_ch(values=e_cat, names=sample.E.names)
    return z_prev, p_out, x_out, e_out


# ---------------------------------------------------------------------------
# Test 1: parameter shapes
# ---------------------------------------------------------------------------

def test_fit_parameter_shapes():
    """fit() produces A(10,4), B(10,6), C(10,1), d(10,), Σ(10,10) symmetric PSD."""
    cohort = _make_cohort()
    z_prev, p, x, e = _concat_channels(cohort)
    model = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)

    assert model.A.shape == (10, 4), f"A shape {model.A.shape}"
    assert model.B.shape == (10, 6), f"B shape {model.B.shape}"
    assert model.C.shape == (10, 1), f"C shape {model.C.shape}"
    assert model.d.shape == (10,), f"d shape {model.d.shape}"
    assert model.Σ.shape == (10, 10), f"Σ shape {model.Σ.shape}"

    # Symmetry.
    np.testing.assert_allclose(model.Σ, model.Σ.T, atol=1e-10)

    # PSD: all eigenvalues >= 0.
    eigvals = np.linalg.eigvalsh(model.Σ)
    assert np.all(eigvals >= -1e-10), f"Σ not PSD; min eigval = {eigvals.min():.3e}"


# ---------------------------------------------------------------------------
# Test 2: forward round-trip shape and names
# ---------------------------------------------------------------------------

def test_forward_roundtrip():
    """forward() returns X with correct shape and names. Does not test values == training X."""
    cohort = _make_cohort()
    z_prev, p, x, e = _concat_channels(cohort)
    model = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)

    x_pred = model.forward(z_prev, p, e)

    T = p.values.shape[0]
    assert x_pred.values.shape == (T, 10), f"values shape {x_pred.values.shape}"
    assert x_pred.names == _X_NAMES, f"names mismatch: {x_pred.names}"
    assert x_pred.is_rest.shape == (T,), f"is_rest shape {x_pred.is_rest.shape}"
    assert not np.any(x_pred.is_rest), "forward should return is_rest=False everywhere"


# ---------------------------------------------------------------------------
# Test 3: inverse recovers held-out component
# ---------------------------------------------------------------------------

def test_inverse_recovers_held_out():
    """inverse() recovers a held-out X component within 5 residual stds.

    Held-out component: best_effort_mean_HR (index 2, passthrough transform).
    Uses first non-rest row of the first athlete as the test row.
    """
    cohort = _make_cohort()
    z_prev, p, x, e = _concat_channels(cohort)
    model = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)

    # Find residual std for best_effort_mean_HR on training data.
    from statepace.observation import _transform
    x_tilde = _transform(x.values, x.names)
    active = ~x.is_rest
    mu_tilde_train = model._mean_tilde(z_prev, p, e)
    resid = x_tilde[active] - mu_tilde_train[active]
    held_out_name = "best_effort_mean_HR"
    h_idx = list(x.names).index(held_out_name)
    resid_std = float(np.std(resid[:, h_idx]))

    # Pick the first non-rest row.
    row_idx = int(np.argwhere(active)[0, 0])

    # Build a single-row X_partial with held-out component NaN'd.
    from statepace.channels import X as X_ch, P as P_ch, E as E_ch
    x_vals_partial = x.values[row_idx: row_idx + 1, :].copy()
    true_val = float(x_vals_partial[0, h_idx])
    x_vals_partial[0, h_idx] = np.nan

    x_partial = X_ch(
        values=x_vals_partial,
        names=x.names,
        is_rest=x.is_rest[row_idx: row_idx + 1],
    )
    z_row = Z(
        mean=z_prev.mean[row_idx: row_idx + 1, :],
        cov=None,
        dates=z_prev.dates[row_idx: row_idx + 1],
    )
    p_row = P_ch(values=p.values[row_idx: row_idx + 1, :], names=p.names)
    e_row = E_ch(values=e.values[row_idx: row_idx + 1, :], names=e.names)

    spec = ConditioningSpec(
        held_out=held_out_name,
        p_mode="fixed",
        e_mode="fixed",
        z_mode="fixed",
        x_components="fixed",
    )
    recovered = model.inverse(z_row, p_row, e_row, x_partial, spec)

    assert recovered.shape == (1,), f"Expected shape (1,), got {recovered.shape}"
    tol = 5.0 * resid_std
    diff = abs(float(recovered[0]) - true_val)
    assert diff < tol, (
        f"inverse off by {diff:.4f}; tolerance {tol:.4f} (5 × resid_std={resid_std:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: log_prob shape and rest-day zeros
# ---------------------------------------------------------------------------

def test_log_prob_shape_and_rest_zero():
    """log_prob() returns shape (T,); rest rows = 0.0; non-rest rows finite."""
    cohort = _make_cohort()
    z_prev, p, x, e = _concat_channels(cohort)
    model = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)

    # Use a held-out athlete that was not seen in fit to probe OOS behaviour.
    # Since all athletes are concatenated in fit here, use the last athlete's
    # channels individually to get a fresh T.
    last_key = list(cohort.keys())[-1]
    ch = cohort[last_key]
    T = ch.X.values.shape[0]

    rng = np.random.default_rng(_SEED + 99)
    z_mean_ho = rng.standard_normal((T, 4))
    z_ho = Z(mean=z_mean_ho, cov=None, dates=np.zeros(T, dtype="datetime64[ns]"))

    lp = model.log_prob(z_ho, ch.P, ch.E, ch.X)

    assert lp.shape == (T,), f"log_prob shape {lp.shape}"

    rest_mask = ch.X.is_rest
    if np.any(rest_mask):
        np.testing.assert_array_equal(
            lp[rest_mask], 0.0,
            err_msg="log_prob on rest rows must be exactly 0.0",
        )

    non_rest_lp = lp[~rest_mask]
    assert np.all(np.isfinite(non_rest_lp)), (
        f"Non-rest log_prob has non-finite values: {non_rest_lp[~np.isfinite(non_rest_lp)]}"
    )


# ---------------------------------------------------------------------------
# Test 5: fit is deterministic
# ---------------------------------------------------------------------------

def test_fit_deterministic():
    """fit() twice on identical input produces element-wise identical parameters."""
    cohort = _make_cohort()
    z_prev, p, x, e = _concat_channels(cohort)

    model1 = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)
    model2 = RiegelScoreHRStep(d_Z=4).fit(z_prev, p, x, e)

    np.testing.assert_array_equal(model1.A, model2.A)
    np.testing.assert_array_equal(model1.B, model2.B)
    np.testing.assert_array_equal(model1.C, model2.C)
    np.testing.assert_array_equal(model1.d, model2.d)
    np.testing.assert_array_equal(model1.Σ, model2.Σ)
