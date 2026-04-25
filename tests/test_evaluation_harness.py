"""Tests for statepace/evaluation/harness.py: assign_cohorts, make_splits, run_evaluation, run_sweep."""
from __future__ import annotations

import math

import numpy as np
import pytest

from statepace.evaluation.harness import (
    EvalResult,
    EvalSplit,
    FormsBundle,
    XPredictive,
    assign_cohorts,
    make_splits,
    run_evaluation,
    run_sweep,
)
from tests.fixtures.synthetic import make_m2_test_cohort, make_channels

# Hyperparameters per D2 / plan §M2.
WARMUP_DAYS = 90
TRAIN_DAYS = 210
TEST_DAYS = 60
VOLUME_BUCKET_EDGES = [100.0]
VOLUME_COMPONENT = "dist_km"


def test_assign_cohorts_deterministic_given_seed():
    cohort, meta = make_m2_test_cohort()

    a1 = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=42,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )
    a2 = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=42,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )
    assert a1 == a2

    a3 = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=99,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )
    # At least one athlete must differ between the two seeds.
    assert any(a1[sid] != a3[sid] for sid in a1)


def test_assign_cohorts_per_stratum_minimum():
    cohort, meta = make_m2_test_cohort()

    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=0,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    # Stratum sizes: F-low=19, F-high=19, M-low=6, M-high=6.
    # ceil(19 * 0.10) = 2, ceil(6 * 0.10) = 1.
    expected_min = {
        ("F", 0): 2,   # F-low, below edge 100.0
        ("F", 1): 2,   # F-high, above edge 100.0
        ("M", 0): 1,   # M-low
        ("M", 1): 1,   # M-high
    }

    # Count validation per (sex, bucket).
    LOW_SUM = 0.3 * TRAIN_DAYS   # ~63.0
    HIGH_SUM = 0.7 * TRAIN_DAYS  # ~147.0

    def bucket(sid: str) -> int:
        return int(np.digitize(LOW_SUM if int(sid[2:]) < (19 if sid[0] == "F" else 6) else HIGH_SUM, VOLUME_BUCKET_EDGES))

    counts: dict[tuple[str, int], int] = {}
    for sid, label in assignment.items():
        if label == "validation":
            sex = sid[0]
            b = bucket(sid)
            key = (sex, b)
            counts[key] = counts.get(key, 0) + 1

    total_validation = sum(counts.values())
    assert total_validation == 6  # 2+2+1+1

    for key, expected in expected_min.items():
        assert counts.get(key, 0) >= expected


def test_assign_cohorts_no_overlap():
    cohort, meta = make_m2_test_cohort()

    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=7,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    train_ids = {sid for sid, label in assignment.items() if label == "train"}
    val_ids = {sid for sid, label in assignment.items() if label == "validation"}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(cohort.keys())


def test_assign_cohorts_volume_component_missing_raises():
    cohort, meta = make_m2_test_cohort()

    with pytest.raises(ValueError, match="volume_component"):
        assign_cohorts(
            cohort,
            meta,
            validation_fraction=0.10,
            seed=0,
            volume_bucket_edges=VOLUME_BUCKET_EDGES,
            warmup_days=WARMUP_DAYS,
            train_days=TRAIN_DAYS,
            volume_component="nonexistent_channel",
        )


def test_assign_cohorts_key_mismatch_raises():
    cohort, meta = make_m2_test_cohort()

    # Drop one athlete from meta to create a mismatch.
    bad_meta = {k: v for k, v in meta.items() if k != "F_00"}

    with pytest.raises(ValueError):
        assign_cohorts(
            cohort,
            bad_meta,
            validation_fraction=0.10,
            seed=0,
            volume_bucket_edges=VOLUME_BUCKET_EDGES,
            warmup_days=WARMUP_DAYS,
            train_days=TRAIN_DAYS,
            volume_component=VOLUME_COMPONENT,
        )


def test_make_splits_shapes():
    cohort, meta = make_m2_test_cohort()
    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=0,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    splits = list(make_splits(
        cohort,
        meta,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        cohort_assignment=assignment,
    ))

    # seed=0 → 44 train, 6 val → 2*44 + 6 = 94 splits total
    n_train_athletes = sum(1 for v in assignment.values() if v == "train")
    n_val_athletes = sum(1 for v in assignment.values() if v == "validation")
    assert len(splits) == 2 * n_train_athletes + n_val_athletes

    train_splits = [sp for sp in splits if sp.cohort == "train"]
    test_splits = [sp for sp in splits if sp.cohort == "test"]
    val_splits = [sp for sp in splits if sp.cohort == "validation"]

    assert len(train_splits) == n_train_athletes
    assert len(test_splits) == n_train_athletes
    assert len(val_splits) == n_val_athletes

    for sp in splits:
        assert sp.fit_idx.shape == (WARMUP_DAYS + TRAIN_DAYS,)
        assert sp.fit_idx.dtype == int
        assert sp.score_idx.dtype == int

    for sp in train_splits:
        assert sp.score_idx.shape == (TRAIN_DAYS,)

    for sp in test_splits + val_splits:
        assert sp.score_idx.shape == (TEST_DAYS,)


def test_make_splits_no_overlap():
    cohort, meta = make_m2_test_cohort()
    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=0,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    splits = list(make_splits(
        cohort,
        meta,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        cohort_assignment=assignment,
    ))

    full_range = set(range(WARMUP_DAYS + TRAIN_DAYS + TEST_DAYS))
    for sp in splits:
        fit_set = set(sp.fit_idx.tolist())
        score_set = set(sp.score_idx.tolist())
        if sp.cohort == "train":
            # In-sample: score_idx is a subset of fit_idx
            assert score_set <= fit_set
        else:
            # "test" and "validation": disjoint, union covers full timeline
            assert fit_set.isdisjoint(score_set)
            assert fit_set | score_set == full_range


def test_make_splits_cohort_label():
    cohort, meta = make_m2_test_cohort()
    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=0,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    splits = list(make_splits(
        cohort,
        meta,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        cohort_assignment=assignment,
    ))

    from collections import defaultdict
    labels_by_sid: dict[str, list[str]] = defaultdict(list)
    for sp in splits:
        labels_by_sid[sp.subject_id].append(sp.cohort)

    for sid, labels in labels_by_sid.items():
        if assignment[sid] == "train":
            assert sorted(labels) == ["test", "train"]
        else:
            assert labels == ["validation"]


def test_make_splits_short_timeline_raises():
    """An athlete with fewer than 360 days triggers ValueError."""
    cohort, meta = make_m2_test_cohort()
    assignment = assign_cohorts(
        cohort,
        meta,
        validation_fraction=0.10,
        seed=0,
        volume_bucket_edges=VOLUME_BUCKET_EDGES,
        warmup_days=WARMUP_DAYS,
        train_days=TRAIN_DAYS,
        volume_component=VOLUME_COMPONENT,
    )

    # Replace one athlete with a short timeline (T=100 days).
    short_dates = np.arange("2024-01-01", "2024-04-10", dtype="datetime64[D]").astype("datetime64[ns]")
    T_short = len(short_dates)
    short_channels = make_channels(
        subject_id="F_00",
        dates=short_dates,
        P_values=np.ones((T_short, 1), dtype=float) * 0.3,
        P_names=("dist_km",),
        X_values=np.ones((T_short, 1), dtype=float),
        X_names=("pace_s_km",),
        is_rest=np.zeros(T_short, dtype=bool),
        E_values=np.ones((T_short, 1), dtype=float),
        E_names=("temp_c",),
    )
    bad_cohort = {**cohort, "F_00": short_channels}

    with pytest.raises(ValueError, match="F_00"):
        list(make_splits(
            bad_cohort,
            meta,
            warmup_days=WARMUP_DAYS,
            train_days=TRAIN_DAYS,
            test_days=TEST_DAYS,
            cohort_assignment=assignment,
        ))


# ---------------------------------------------------------------------------
# Helpers shared across run_evaluation / run_sweep tests
# ---------------------------------------------------------------------------

def _make_canonical_bundle(d_Z: int = 4, max_iterations: int = 20) -> FormsBundle:
    """Build the canonical M6 wiring: riegel-score-hrstep + linear-gaussian + joint-mle-kalman."""
    from statepace.filter import JointMLEKalman, JointMLEKalmanConfig, Prior
    from statepace.observation import RiegelScoreHRStep
    from statepace.transitions import LinearGaussianRestTransition, LinearGaussianWorkoutTransition

    cfg = JointMLEKalmanConfig(
        d_Z=d_Z,
        tau=(1.0, 7.0, 28.0, 84.0),
        max_iterations=max_iterations,
        n_seeds=1,
        patience=5,
        eval_every=5,
    )
    prior = Prior(d_Z=d_Z, diffuse=True, mean=None, cov=None)
    return FormsBundle(
        label="canonical",
        observation=RiegelScoreHRStep(d_Z=d_Z),
        workout_transition=LinearGaussianWorkoutTransition(d_Z=d_Z),
        rest_transition=LinearGaussianRestTransition(d_Z=d_Z, max_consecutive_rest_days=10),
        estimator=JointMLEKalman(cfg=cfg),
        prior=prior,
    )


def _make_small_cohort_and_splits(
    n_athletes: int = 4,
    n_days: int = 60,
    seed: int = 0,
    warmup_days: int = 10,
    train_days: int = 30,
    score_days: int = 20,
) -> tuple:
    """Return (cohort, splits_list) for run_evaluation tests."""
    from tests.fixtures.reference_impls.joint_mle_kalman import make_joint_mle_kalman_cohort

    cohort, _true_mu0, _theta = make_joint_mle_kalman_cohort(
        n_athletes=n_athletes,
        n_days=n_days,
        d_Z=4,
        tau=(1.0, 7.0, 28.0, 84.0),
        sigma0_sq=1.0,
        seed=seed,
    )

    fit_idx = np.arange(0, warmup_days + train_days, dtype=int)
    score_idx = np.arange(warmup_days + train_days, warmup_days + train_days + score_days, dtype=int)

    subject_ids = list(cohort.keys())
    # First athlete is "train", rest are "test" (simple; validates both cohort paths).
    splits = []
    for i, sid in enumerate(subject_ids):
        if i == 0:
            splits.append(EvalSplit(subject_id=sid, cohort="train", fit_idx=fit_idx, score_idx=fit_idx[warmup_days:]))
        else:
            splits.append(EvalSplit(subject_id=sid, cohort="test", fit_idx=fit_idx, score_idx=score_idx))

    return cohort, splits


# ---------------------------------------------------------------------------
# run_evaluation tests
# ---------------------------------------------------------------------------

def test_run_evaluation_returns_eval_result():
    """run_evaluation returns an EvalResult with correct keys."""
    from statepace.filter import GaussianZPosterior

    cohort, splits = _make_small_cohort_and_splits()
    bundle = _make_canonical_bundle()

    result = run_evaluation(
        cohort=cohort,
        splits=splits,
        estimator=bundle.estimator,
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        prior=bundle.prior,
        n_samples=10,
        rng=np.random.default_rng(1),
    )

    assert isinstance(result, EvalResult)
    assert set(result.Z_hat.keys()) == set(cohort.keys())
    assert set(result.X_pred.keys()) == set(cohort.keys())
    assert set(result.cohort.keys()) == set(cohort.keys())
    assert set(result.rest_bound_violations.keys()) == set(cohort.keys())

    for sid in cohort:
        assert isinstance(result.Z_hat[sid], GaussianZPosterior)


def test_run_evaluation_x_pred_shapes():
    """XPredictive.samples has shape (n_samples, T_score, d_X) for each athlete."""
    n_samples = 10
    cohort, splits = _make_small_cohort_and_splits(
        warmup_days=10, train_days=30, score_days=20
    )
    bundle = _make_canonical_bundle()

    result = run_evaluation(
        cohort=cohort,
        splits=splits,
        estimator=bundle.estimator,
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        prior=bundle.prior,
        n_samples=n_samples,
        rng=np.random.default_rng(2),
    )

    d_X = 10  # riegel-score-hrstep has 10 X channels
    for sid, xp in result.X_pred.items():
        assert isinstance(xp, XPredictive)
        assert xp.n_samples == n_samples
        assert xp.samples.ndim == 3
        assert xp.samples.shape[0] == n_samples
        assert xp.samples.shape[2] == d_X


def test_run_evaluation_rest_bound_violations_bool_shape():
    """rest_bound_violations is a boolean array aligned with the score window."""
    cohort, splits = _make_small_cohort_and_splits()
    bundle = _make_canonical_bundle()

    result = run_evaluation(
        cohort=cohort,
        splits=splits,
        estimator=bundle.estimator,
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        prior=bundle.prior,
        n_samples=5,
        rng=np.random.default_rng(3),
    )

    for sid, violations in result.rest_bound_violations.items():
        assert violations.dtype == bool
        # Shape should match the score_idx for this subject's split.
        # Find the split used for this subject.
        xp = result.X_pred[sid]
        T_score = xp.samples.shape[1]
        assert violations.shape == (T_score,)


def test_run_evaluation_cohort_mapping_complete():
    """cohort mapping covers all athletes with valid cohort labels."""
    cohort, splits = _make_small_cohort_and_splits()
    bundle = _make_canonical_bundle()

    result = run_evaluation(
        cohort=cohort,
        splits=splits,
        estimator=bundle.estimator,
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        prior=bundle.prior,
        n_samples=5,
        rng=np.random.default_rng(4),
    )

    valid_labels = {"train", "test", "validation"}
    for sid, label in result.cohort.items():
        assert label in valid_labels
    assert set(result.cohort.keys()) == set(cohort.keys())


def test_run_evaluation_n_samples_configurable():
    """n_samples is reflected in XPredictive.n_samples and samples.shape[0]."""
    for n in (5, 15):
        cohort, splits = _make_small_cohort_and_splits()
        bundle = _make_canonical_bundle()

        result = run_evaluation(
            cohort=cohort,
            splits=splits,
            estimator=bundle.estimator,
            observation=bundle.observation,
            workout_transition=bundle.workout_transition,
            rest_transition=bundle.rest_transition,
            prior=bundle.prior,
            n_samples=n,
            rng=np.random.default_rng(5),
        )

        for sid, xp in result.X_pred.items():
            assert xp.n_samples == n
            assert xp.samples.shape[0] == n


# ---------------------------------------------------------------------------
# run_sweep tests
# ---------------------------------------------------------------------------

def test_run_sweep_two_bundles():
    """run_sweep returns SweepResult with both bundle labels as keys."""
    from statepace.filter import JointMLEKalman, JointMLEKalmanConfig, Prior
    from statepace.observation import RiegelScoreHRStep
    from statepace.transitions import LinearGaussianRestTransition, LinearGaussianWorkoutTransition

    cohort, splits = _make_small_cohort_and_splits()

    d_Z = 4
    cfg = JointMLEKalmanConfig(
        d_Z=d_Z, tau=(1.0, 7.0, 28.0, 84.0),
        max_iterations=10, n_seeds=1, patience=3, eval_every=3,
    )
    prior = Prior(d_Z=d_Z, diffuse=True, mean=None, cov=None)

    bundle_a = FormsBundle(
        label="bundle_a",
        observation=RiegelScoreHRStep(d_Z=d_Z),
        workout_transition=LinearGaussianWorkoutTransition(d_Z=d_Z),
        rest_transition=LinearGaussianRestTransition(d_Z=d_Z, max_consecutive_rest_days=10),
        estimator=JointMLEKalman(cfg=cfg),
        prior=prior,
    )
    bundle_b = FormsBundle(
        label="bundle_b",
        observation=RiegelScoreHRStep(d_Z=d_Z),
        workout_transition=LinearGaussianWorkoutTransition(d_Z=d_Z),
        rest_transition=LinearGaussianRestTransition(d_Z=d_Z, max_consecutive_rest_days=10),
        estimator=JointMLEKalman(cfg=cfg),
        prior=prior,
    )

    sweep = run_sweep(cohort=cohort, splits=splits, bundles=[bundle_a, bundle_b], n_samples=5)

    assert set(sweep.results.keys()) == {"bundle_a", "bundle_b"}
    assert set(sweep.bundles.keys()) == {"bundle_a", "bundle_b"}
    for label in ("bundle_a", "bundle_b"):
        assert isinstance(sweep.results[label], EvalResult)


def test_run_sweep_duplicate_labels_raises():
    """Duplicate bundle labels raise ValueError."""
    cohort, splits = _make_small_cohort_and_splits()
    bundle = _make_canonical_bundle()
    bundle2 = FormsBundle(
        label="canonical",   # same label
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        estimator=bundle.estimator,
        prior=bundle.prior,
    )

    with pytest.raises(ValueError, match="canonical"):
        run_sweep(cohort=cohort, splits=splits, bundles=[bundle, bundle2], n_samples=5)


def test_run_evaluation_canonical_smoke():
    """Full canonical wiring (riegel-score-hrstep + linear-gaussian + joint-mle-kalman) on a tiny cohort runs without error."""
    cohort, splits = _make_small_cohort_and_splits(n_athletes=4, n_days=60)
    bundle = _make_canonical_bundle(max_iterations=15)

    result = run_evaluation(
        cohort=cohort,
        splits=splits,
        estimator=bundle.estimator,
        observation=bundle.observation,
        workout_transition=bundle.workout_transition,
        rest_transition=bundle.rest_transition,
        prior=bundle.prior,
        n_samples=10,
        rng=np.random.default_rng(42),
    )

    assert isinstance(result, EvalResult)
    assert len(result.Z_hat) == len(cohort)
    assert len(result.X_pred) == len(cohort)
    assert len(result.cohort) == len(cohort)
    assert len(result.rest_bound_violations) == len(cohort)
