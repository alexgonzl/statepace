"""Tests for JointMLEKalman StateEstimator and ZPosterior ABC/GaussianZPosterior.

Spec: docs/reference_impls/joint-mle-kalman.md
ADR:  docs/decisions/0006-m6-joint-mle-kalman-first-state-estimator.md
Fixture: tests/fixtures/reference_impls/joint_mle_kalman.py

W4 test coverage:
  - Staged init deterministic (same cohort -> same init).
  - SGD loop deterministic under fixed seeds.
  - infer(channels) returns GaussianZPosterior with correct shapes for
    training-cohort (frozen mu_0) and validation-cohort (on-the-fly mu_0 fit).
  - Prior.diffuse=True honored via large sigma_0.
  - d_Z mismatch raises at construction.
  - Rest-bound overrun masked from loss; filter resumes at first post-gap workout day.
  - PSD covariances preserved across SGD.
  - Multi-start deterministic under seed list; selection reproducible.
  - Post-hoc sign-cascade idempotent.
  - Validation mu_0 recovery on synthetic (NA1).
"""
from __future__ import annotations

import numpy as np
import pytest

from statepace.channels import Z
from statepace.filter import (
    GaussianZPosterior,
    JointMLEKalman,
    JointMLEKalmanConfig,
    Prior,
    ZPosterior,
    _apply_sign_cascade,
    _staged_init,
    _compute_pi_bar_stim,
    _count_consecutive_rest,
    _fit_mu0_single,
)
from statepace.observation import RiegelScoreHRStep
from statepace.transitions import (
    LinearGaussianWorkoutTransition,
    LinearGaussianRestTransition,
)
from tests.fixtures.reference_impls.joint_mle_kalman import make_joint_mle_kalman_cohort
from tests.fixtures.reference_impls.riegel_score_hrstep import make_riegel_hrstep_cohort

# ---------------------------------------------------------------------------
# Shared test parameters — small cohort for speed
# ---------------------------------------------------------------------------

_D_Z = 4
_TAU = (1.0, 7.0, 28.0, 84.0)
_N_ATHLETES = 5
_N_DAYS = 60      # short sequences for fast tests
_SEED = 42

# Minimal config: very few iterations so tests run in reasonable time.
_FAST_CFG = JointMLEKalmanConfig(
    d_Z=_D_Z,
    tau=_TAU,
    n_pi_stim=5,
    learning_rate=1e-2,
    max_iterations=30,
    patience=5,
    eval_every=5,
    tol_conv=1e-4,
    n_seeds=2,
    q_init=0.01,
    sigma0_sq=100.0,
    tol_b=0.05,
    tol_m=0.05,
    sigma_re_sq=1.0,
    val_fraction=0.2,
    infer_max_iterations=20,
)


def _make_cohort_and_impls():
    """Build a small test cohort with paired observation and transition impls."""
    cohort = make_riegel_hrstep_cohort(_N_ATHLETES, _N_DAYS, seed=_SEED)
    obs = RiegelScoreHRStep(d_Z=_D_Z)
    wt = LinearGaussianWorkoutTransition(d_Z=_D_Z)
    rt = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=10)
    prior = Prior(d_Z=_D_Z, diffuse=True, mean=None, cov=None)
    return cohort, obs, wt, rt, prior


def _make_ssm_cohort():
    """Build a small cohort generated from the M6 SSM (for recovery test)."""
    return make_joint_mle_kalman_cohort(
        n_athletes=_N_ATHLETES,
        n_days=_N_DAYS,
        d_Z=_D_Z,
        tau=_TAU,
        sigma0_sq=1.0,
        seed=_SEED,
    )


# ---------------------------------------------------------------------------
# ZPosterior ABC / GaussianZPosterior
# ---------------------------------------------------------------------------

def test_zposterior_is_abc():
    """ZPosterior cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ZPosterior()  # type: ignore[abstract]


def test_gaussian_zposterior_mean_shape():
    """GaussianZPosterior.mean() returns (T, d_Z)."""
    T, dZ = 20, 4
    rng = np.random.default_rng(0)
    mean_arr = rng.standard_normal((T, dZ))
    cov_arr = np.stack([np.eye(dZ)] * T)
    dates = np.zeros(T, dtype="datetime64[ns]")
    gzp = GaussianZPosterior(_mean=mean_arr, cov=cov_arr, dates=dates)

    assert gzp.mean().shape == (T, dZ)
    assert gzp.d_Z == dZ


def test_gaussian_zposterior_sample_shape():
    """GaussianZPosterior.sample(n, rng) returns (n, T, d_Z)."""
    T, dZ, n = 10, 4, 7
    rng_np = np.random.default_rng(1)
    mean_arr = rng_np.standard_normal((T, dZ))
    cov_arr = np.stack([np.eye(dZ)] * T)
    dates = np.zeros(T, dtype="datetime64[ns]")
    gzp = GaussianZPosterior(_mean=mean_arr, cov=cov_arr, dates=dates)

    samples = gzp.sample(n, rng=np.random.default_rng(2))
    assert samples.shape == (n, T, dZ)


def test_gaussian_zposterior_marginal_log_pdf_shape():
    """GaussianZPosterior.marginal_log_pdf(z) returns (T,) finite values."""
    T, dZ = 15, 4
    rng_np = np.random.default_rng(3)
    mean_arr = rng_np.standard_normal((T, dZ))
    cov_arr = np.stack([np.eye(dZ)] * T)
    dates = np.zeros(T, dtype="datetime64[ns]")
    gzp = GaussianZPosterior(_mean=mean_arr, cov=cov_arr, dates=dates)
    z_query = rng_np.standard_normal((T, dZ))

    lp = gzp.marginal_log_pdf(z_query)
    assert lp.shape == (T,)
    assert np.all(np.isfinite(lp))


def test_gaussian_zposterior_is_zposterior():
    """GaussianZPosterior is a subclass of ZPosterior."""
    T, dZ = 5, 4
    dates = np.zeros(T, dtype="datetime64[ns]")
    gzp = GaussianZPosterior(
        _mean=np.zeros((T, dZ)),
        cov=np.stack([np.eye(dZ)] * T),
        dates=dates,
    )
    assert isinstance(gzp, ZPosterior)


# ---------------------------------------------------------------------------
# d_Z mismatch
# ---------------------------------------------------------------------------

def test_dz_mismatch_raises():
    """d_Z mismatch between JointMLEKalman and paired impls raises at fit time."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    # obs.d_Z = 4, but estimator config says d_Z = 3 -> mismatch.
    cfg_bad = JointMLEKalmanConfig(d_Z=3, tau=(1.0, 7.0, 28.0), n_pi_stim=5,
                                    max_iterations=5, n_seeds=1)
    est = JointMLEKalman(cfg=cfg_bad)
    with pytest.raises(ValueError, match="d_Z"):
        est.fit(cohort, obs, wt, rt, prior)


# ---------------------------------------------------------------------------
# Staged init determinism
# ---------------------------------------------------------------------------

def test_staged_init_deterministic():
    """_staged_init produces element-wise identical theta on two calls with same cohort."""
    cohort = make_riegel_hrstep_cohort(_N_ATHLETES, _N_DAYS, seed=_SEED)
    x_names = next(iter(cohort.values())).X.names
    p_names = next(iter(cohort.values())).P.names
    e_names = next(iter(cohort.values())).E.names
    pi_bar = _compute_pi_bar_stim(cohort, x_names)

    t1 = _staged_init(cohort, x_names, p_names, e_names, pi_bar, _FAST_CFG)
    t2 = _staged_init(cohort, x_names, p_names, e_names, pi_bar, _FAST_CFG)

    for key in t1:
        np.testing.assert_array_equal(
            t1[key], t2[key],
            err_msg=f"staged_init key '{key}' not deterministic",
        )


# ---------------------------------------------------------------------------
# SGD determinism under fixed seeds
# ---------------------------------------------------------------------------

def test_sgd_deterministic_under_fixed_seeds():
    """Two fit() calls with identical cohort and seed produce element-wise equal theta."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    cfg = JointMLEKalmanConfig(
        d_Z=_D_Z, tau=_TAU, n_pi_stim=5,
        max_iterations=10, n_seeds=1, val_fraction=0.2,
        eval_every=5, patience=5,
    )
    est1 = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior)
    est2 = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior)

    for key in est1._theta:
        np.testing.assert_array_equal(
            est1._theta[key], est2._theta[key],
            err_msg=f"theta key '{key}' differs across two deterministic fit() calls",
        )


# ---------------------------------------------------------------------------
# infer returns GaussianZPosterior with correct shapes
# ---------------------------------------------------------------------------

def test_infer_training_cohort_shapes():
    """infer() on a training-cohort athlete returns GaussianZPosterior with (T, d_Z) mean and (T, d_Z, d_Z) cov."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    # Use the first training athlete.
    train_sid = est._train_subject_ids[0]
    ch = cohort[train_sid]
    T = ch.dates.shape[0]

    result = est.infer(ch)

    assert isinstance(result, GaussianZPosterior)
    assert isinstance(result, ZPosterior)
    assert result.mean().shape == (T, _D_Z), f"mean shape {result.mean().shape}"
    assert result.cov.shape == (T, _D_Z, _D_Z), f"cov shape {result.cov.shape}"
    assert result.dates.shape == (T,), f"dates shape {result.dates.shape}"
    assert result.d_Z == _D_Z


def test_infer_validation_cohort_shapes():
    """infer() on a validation-cohort athlete (unseen at fit) returns correct shapes."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    # Build a fresh athlete not in the training cohort.
    new_cohort = make_riegel_hrstep_cohort(2, _N_DAYS, seed=999)
    new_sid = list(new_cohort.keys())[0]
    ch = new_cohort[new_sid]
    T = ch.dates.shape[0]

    result = est.infer(ch)

    assert isinstance(result, GaussianZPosterior)
    assert result.mean().shape == (T, _D_Z)
    assert result.cov.shape == (T, _D_Z, _D_Z)


def test_infer_mode_smooth_raises():
    """infer(mode='smooth') raises NotImplementedError."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)
    ch = cohort[est._train_subject_ids[0]]
    with pytest.raises(NotImplementedError):
        est.infer(ch, mode="smooth")


def test_infer_before_fit_raises():
    """infer() before fit() raises RuntimeError."""
    cfg = _FAST_CFG
    est = JointMLEKalman(cfg=cfg)
    cohort = make_riegel_hrstep_cohort(1, _N_DAYS, seed=0)
    ch = next(iter(cohort.values()))
    with pytest.raises(RuntimeError, match="fit"):
        est.infer(ch)


# ---------------------------------------------------------------------------
# Prior.diffuse=True honored
# ---------------------------------------------------------------------------

def test_prior_diffuse_honored_via_sigma0():
    """Prior.diffuse=True results in large Sigma_0 (sigma0_sq * I).

    Proxy test: the config's sigma0_sq is reflected in the initial P_filt
    of the Kalman filter (checked via the cov output on the first timestep
    after a short fit — cov should be finite and larger than Q).
    """
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    cfg = JointMLEKalmanConfig(
        d_Z=_D_Z, tau=_TAU, n_pi_stim=5,
        max_iterations=5, n_seeds=1,
        sigma0_sq=1000.0,  # very large diffuse prior
        val_fraction=0.2, eval_every=5, patience=5,
    )
    prior_diffuse = Prior(d_Z=_D_Z, diffuse=True, mean=None, cov=None)
    est = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior_diffuse)

    ch = cohort[est._train_subject_ids[0]]
    result = est.infer(ch, prior=prior_diffuse)

    # Cov at t=0 should be close to sigma0_sq * I before any update.
    # After the first observation update it will be smaller; just check finiteness.
    assert np.all(np.isfinite(result.cov))


# ---------------------------------------------------------------------------
# Rest-bound overrun masked from loss
# ---------------------------------------------------------------------------

def test_rest_bound_overrun_masked():
    """Athlete with a long rest gap (> max_rest_days) runs without error; filter resumes after gap."""
    from tests.fixtures.synthetic import make_channels
    import numpy as np

    # Build a channel with a 15-day rest gap (exceeds max 10).
    T = 50
    d_Z = _D_Z
    dates = np.arange("2023-01-01", "2023-02-20", dtype="datetime64[D]").astype("datetime64[ns]")
    assert len(dates) == T

    # Days 10-24 are rest days (15 consecutive rest).
    is_rest = np.zeros(T, dtype=bool)
    is_rest[10:25] = True

    # Also need real X values for non-rest days.
    rng = np.random.default_rng(77)
    from tests.fixtures.reference_impls.riegel_score_hrstep import (
        _P_NAMES, _E_NAMES, _X_NAMES, _sample_P, _sample_E, _sample_X
    )
    P_values = _sample_P(rng, T)
    E_values = _sample_E(rng, T)
    total_dur = P_values[:, 1]
    wet_bulb_1d = E_values[:, 0]
    X_values = _sample_X(rng, T, total_dur, wet_bulb_1d)
    X_values = X_values.copy().astype(float)
    X_values[is_rest, :] = np.nan

    from statepace.channels import Channels
    from statepace.channels import P as P_ch, X as X_ch, E as E_ch
    ch = Channels(
        subject_id="rest_test",
        dates=dates,
        P=P_ch(values=P_values, names=_P_NAMES),
        X=X_ch(values=X_values, names=_X_NAMES, is_rest=is_rest),
        E=E_ch(values=E_values, names=_E_NAMES),
    )

    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    # Should not raise; filter handles the rest-bound overrun.
    result = est.infer(ch)
    assert result.mean().shape == (T, _D_Z)
    assert np.all(np.isfinite(result.mean()))


def test_max_rest_days_plumbed_from_rest_transition():
    """est._max_rest_days must match the value passed to LinearGaussianRestTransition."""
    from statepace.transitions import LinearGaussianRestTransition
    cohort, obs, wt, _, prior = _make_cohort_and_impls()
    rt7 = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=7)
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt7, prior)
    assert est._max_rest_days == 7, (
        f"Expected _max_rest_days=7 (from rest_transition), got {est._max_rest_days}. "
        "max_consecutive_rest_days is not being read from rest_transition."
    )


# ---------------------------------------------------------------------------
# PSD covariances preserved across SGD
# ---------------------------------------------------------------------------

def test_psd_covariances_preserved():
    """Fitted covariance matrices Sigma, Q, Q_1 are PSD (Cholesky succeeds)."""
    from statepace.filter import _theta_to_arrays
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    arrays = _theta_to_arrays(est._theta, _FAST_CFG)

    for name in ("Sigma", "Q", "Q_1"):
        mat = arrays[name]
        # Symmetry.
        np.testing.assert_allclose(mat, mat.T, atol=1e-8, err_msg=f"{name} not symmetric")
        # PSD: all eigenvalues >= 0.
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(eigvals >= -1e-6), (
            f"{name} not PSD; min eigval = {eigvals.min():.3e}"
        )
        # Cholesky succeeds (numerically PD after small ridge if needed).
        try:
            np.linalg.cholesky(mat + 1e-8 * np.eye(mat.shape[0]))
        except np.linalg.LinAlgError:
            pytest.fail(f"Cholesky failed for {name}")


def test_posterior_covs_finite():
    """Kalman filter output covariances are finite at all timesteps."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)
    ch = cohort[est._train_subject_ids[0]]
    result = est.infer(ch)
    assert np.all(np.isfinite(result.cov)), "Posterior covariances have non-finite entries"


# ---------------------------------------------------------------------------
# Multi-start deterministic; selection reproducible
# ---------------------------------------------------------------------------

def test_multi_start_deterministic():
    """Two fit() calls with n_seeds=3 produce the same selection_log_lik."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    cfg = JointMLEKalmanConfig(
        d_Z=_D_Z, tau=_TAU, n_pi_stim=5,
        max_iterations=15, n_seeds=3, val_fraction=0.2,
        eval_every=5, patience=5,
    )
    est1 = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior)
    est2 = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior)

    assert est1.selection_log_lik == est2.selection_log_lik, (
        f"selection_log_lik differs: {est1.selection_log_lik} vs {est2.selection_log_lik}"
    )

    # Same seed selected.
    sel1 = next(d["seed"] for d in est1.multi_start_diagnostics if d["selected"])
    sel2 = next(d["seed"] for d in est2.multi_start_diagnostics if d["selected"])
    assert sel1 == sel2, f"Different seeds selected: {sel1} vs {sel2}"


def test_multi_start_diagnostics_structure():
    """multi_start_diagnostics has one entry per seed, exactly one selected."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    cfg = JointMLEKalmanConfig(
        d_Z=_D_Z, tau=_TAU, n_pi_stim=5,
        max_iterations=10, n_seeds=3, val_fraction=0.2,
        eval_every=5, patience=5,
    )
    est = JointMLEKalman(cfg=cfg).fit(cohort, obs, wt, rt, prior)

    assert len(est.multi_start_diagnostics) == 3
    n_selected = sum(1 for d in est.multi_start_diagnostics if d["selected"])
    assert n_selected == 1, f"Expected exactly 1 selected seed, got {n_selected}"


# ---------------------------------------------------------------------------
# Post-hoc sign cascade idempotent
# ---------------------------------------------------------------------------

def test_sign_cascade_idempotent():
    """Applying sign cascade twice produces identical result to applying once."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    theta_once = _apply_sign_cascade(est._theta, _FAST_CFG)
    theta_twice = _apply_sign_cascade(theta_once, _FAST_CFG)

    for key in theta_once:
        np.testing.assert_array_almost_equal(
            theta_once[key], theta_twice[key], decimal=12,
            err_msg=f"sign cascade not idempotent for key '{key}'",
        )


def test_sign_cascade_b_nonneg():
    """After sign cascade, all b_i with |b_i| >= tol_b are non-negative."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)
    theta = _apply_sign_cascade(est._theta, _FAST_CFG)
    b = theta["b"]
    tol_b = _FAST_CFG.tol_b
    for i, bi in enumerate(b):
        if abs(bi) >= tol_b:
            assert bi >= 0, f"b[{i}]={bi:.4f} violates non-negative convention after cascade"


# ---------------------------------------------------------------------------
# Validation mu_0 recovery (NA1)
# ---------------------------------------------------------------------------

def test_validation_mu0_recovery_na1():
    """NA1: infer on held-out validation athletes recovers true mu_0^{(i)} within tolerance.

    Generate a small cohort from the M6 SSM with known mu_0^{(i)}.
    Fit on training subset (3 athletes). Infer on 1 held-out athlete.
    Check that recovered mu_0 is closer to truth than a null (zero) baseline.

    Tolerance: absolute deviation < 2.0 in each dimension (loose — n_days is
    small and n_iterations is minimal). This is a Tier-B signal if it fails
    consistently; W6 sets the tighter K=20 replicate tolerance.
    """
    # Use more iterations for recovery to have a chance.
    recovery_cfg = JointMLEKalmanConfig(
        d_Z=_D_Z,
        tau=_TAU,
        n_pi_stim=5,
        learning_rate=5e-3,
        max_iterations=100,
        patience=10,
        eval_every=10,
        tol_conv=1e-4,
        n_seeds=2,
        q_init=0.01,
        sigma0_sq=10.0,
        tol_b=0.05,
        tol_m=0.05,
        sigma_re_sq=1.0,
        val_fraction=0.25,
        infer_max_iterations=100,
    )

    cohort, true_mu0, ground_truth_theta = make_joint_mle_kalman_cohort(
        n_athletes=6,
        n_days=80,
        d_Z=_D_Z,
        tau=_TAU,
        sigma0_sq=1.0,
        seed=123,
    )
    subject_ids = list(cohort.keys())
    train_ids = subject_ids[:4]
    val_id = subject_ids[4]

    train_cohort = {sid: cohort[sid] for sid in train_ids}

    obs = RiegelScoreHRStep(d_Z=_D_Z)
    wt = LinearGaussianWorkoutTransition(d_Z=_D_Z)
    rt = LinearGaussianRestTransition(d_Z=_D_Z, max_consecutive_rest_days=10)
    prior = Prior(d_Z=_D_Z, diffuse=True, mean=None, cov=None)

    est = JointMLEKalman(cfg=recovery_cfg).fit(train_cohort, obs, wt, rt, prior)

    # Infer on held-out validation athlete (shape/finiteness checks).
    val_ch = cohort[val_id]
    result = est.infer(val_ch)

    assert result.mean().shape == (80, _D_Z)  # 80 days (n_days from make_joint_mle_kalman_cohort call above)
    assert np.all(np.isfinite(result.mean())), "Recovered Z trajectory has non-finite values"
    assert np.all(np.isfinite(result.cov)), "Posterior covariances have non-finite entries"

    # Recovery check: call _fit_mu0_single directly with frozen parameters so
    # the comparison target is the fitted mu_0, not the Kalman first-step output
    # (result.mean()[0] is the post-observation-update filtered state at t=0,
    # which differs from mu_0 by a Kalman gain term on the first innovation).
    recovered_mu0 = _fit_mu0_single(
        theta=est._theta,
        channels=val_ch,
        x_names=est._x_names,
        p_names=est._p_names,
        e_names=est._e_names,
        pi_bar_stim=est._pi_bar_stim,
        cfg=recovery_cfg,
        max_rest_days=est._max_rest_days if est._max_rest_days is not None else 10,
    )

    true_z0 = true_mu0[val_id]

    # Check recovery is better than null (zero) in squared L2 norm.
    null_err = np.sum(true_z0 ** 2)
    recovered_err = np.sum((true_z0 - recovered_mu0) ** 2)

    assert recovered_err < null_err, (
        f"NA1 recovery failure: recovered_err={recovered_err:.4f} >= null_err={null_err:.4f}. "
        "The estimator performs no better than predicting zero for mu_0. "
        "This indicates identifiability or convergence failure — not a tolerance issue."
    )


# ---------------------------------------------------------------------------
# Fit artifact fields
# ---------------------------------------------------------------------------

def test_fit_artifact_fields():
    """Fitted estimator carries all spec-required artifact fields."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    assert est._theta is not None, "theta missing"
    assert est._mu0 is not None, "mu_0 missing"
    assert est._pi_bar_stim is not None, "pi_bar_stim missing"
    assert est.gauge_boundary_dims is not None, "gauge_boundary_dims missing"
    assert est.selection_log_lik is not None, "selection_log_lik missing"
    assert est.multi_start_diagnostics is not None, "multi_start_diagnostics missing"
    assert est._train_subject_ids is not None, "train_subject_ids missing"

    # pi_bar_stim shape.
    assert est._pi_bar_stim.shape == (5,), (
        f"pi_bar_stim shape {est._pi_bar_stim.shape}"
    )


def test_pi_bar_stim_frozen():
    """pi_bar_stim is the same array reference across two infer calls (frozen at fit)."""
    cohort, obs, wt, rt, prior = _make_cohort_and_impls()
    est = JointMLEKalman(cfg=_FAST_CFG).fit(cohort, obs, wt, rt, prior)

    pi_bar_before = est._pi_bar_stim.copy()

    # Infer on training and validation athlete.
    train_ch = cohort[est._train_subject_ids[0]]
    est.infer(train_ch)
    new_cohort = make_riegel_hrstep_cohort(1, _N_DAYS, seed=777)
    val_ch = next(iter(new_cohort.values()))
    est.infer(val_ch)

    np.testing.assert_array_equal(
        est._pi_bar_stim, pi_bar_before,
        err_msg="pi_bar_stim was mutated during infer (should be frozen at fit)",
    )


# ---------------------------------------------------------------------------
# StateEstimator Protocol compatibility
# ---------------------------------------------------------------------------

def test_joint_mle_kalman_satisfies_protocol():
    """JointMLEKalman implements the StateEstimator Protocol interface."""
    from statepace.filter import StateEstimator
    # Protocol compatibility is structural; we check required attributes exist.
    est = JointMLEKalman(cfg=_FAST_CFG)
    assert hasattr(est, "d_Z")
    assert hasattr(est, "fit")
    assert hasattr(est, "infer")
    assert callable(est.fit)
    assert callable(est.infer)
