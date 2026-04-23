"""Tests for statepace/evaluation/harness.py: assign_cohorts and make_splits."""
from __future__ import annotations

import math

import numpy as np
import pytest

from statepace.evaluation.harness import assign_cohorts, make_splits
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

    assert len(splits) == 50
    for sp in splits:
        assert sp.fit_idx.shape == (WARMUP_DAYS + TRAIN_DAYS,)  # (300,)
        assert sp.score_idx.shape == (TEST_DAYS,)               # (60,)
        assert sp.fit_idx.dtype == int
        assert sp.score_idx.dtype == int


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

    total = WARMUP_DAYS + TRAIN_DAYS + TEST_DAYS  # 360
    for sp in splits:
        fit_set = set(sp.fit_idx.tolist())
        score_set = set(sp.score_idx.tolist())
        assert fit_set.isdisjoint(score_set)
        assert fit_set | score_set == set(range(total))


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

    for sp in splits:
        assert sp.cohort == assignment[sp.subject_id]


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
