"""Tests for tests/fixtures/synthetic.py::make_channels."""
from __future__ import annotations

import numpy as np
import pytest

from statepace.channels import Channels
from tests.fixtures.synthetic import make_channels

T = 10
REST_IDX = 3


def _dates() -> np.ndarray:
    return np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )


def _is_rest() -> np.ndarray:
    mask = np.zeros(T, dtype=bool)
    mask[REST_IDX] = True
    return mask


def _base_kwargs() -> dict:
    return dict(
        subject_id="athlete_01",
        dates=_dates(),
        P_values=np.ones((T, 2), dtype=float),
        P_names=("dist_km", "elev_m"),
        X_values=np.ones((T, 2), dtype=float) * 2.0,
        X_names=("pace_s_km", "hr_bpm"),
        is_rest=_is_rest(),
        E_values=np.ones((T, 2), dtype=float) * 3.0,
        E_names=("temp_c", "altitude_m"),
    )


def test_happy_path():
    kw = _base_kwargs()
    ch = make_channels(**kw)

    assert isinstance(ch, Channels)
    assert ch.subject_id == "athlete_01"
    assert ch.dates is kw["dates"]
    assert ch.P.values.shape == (T, 2)
    assert ch.X.values.shape == (T, 2)
    assert ch.E.values.shape == (T, 2)
    assert ch.P.names == ("dist_km", "elev_m")
    assert ch.X.names == ("pace_s_km", "hr_bpm")
    assert ch.E.names == ("temp_c", "altitude_m")
    assert np.array_equal(ch.X.is_rest, kw["is_rest"])


def test_nan_on_rest_invariant():
    kw = _base_kwargs()
    # Rest row has non-NaN values — factory must overwrite them.
    kw["X_values"][REST_IDX, :] = 99.0
    ch = make_channels(**kw)

    assert np.all(np.isnan(ch.X.values[REST_IDX, :]))
    # Non-rest rows must be unchanged from what was passed.
    non_rest = np.array([i for i in range(T) if i != REST_IDX])
    np.testing.assert_array_equal(ch.X.values[non_rest, :], kw["X_values"][non_rest, :])


def test_caller_array_not_mutated():
    kw = _base_kwargs()
    original = kw["X_values"].copy()
    make_channels(**kw)
    np.testing.assert_array_equal(kw["X_values"], original)


def test_shape_mismatch_raises():
    kw = _base_kwargs()
    kw["P_values"] = np.ones((T + 1, 2), dtype=float)
    with pytest.raises(ValueError, match="P_values"):
        make_channels(**kw)

    kw2 = _base_kwargs()
    kw2["P_names"] = ("dist_km",)  # len 1 but d_P == 2
    with pytest.raises(ValueError, match="P_names"):
        make_channels(**kw2)


def test_wrong_dtype_raises():
    kw = _base_kwargs()
    kw["is_rest"] = _is_rest().astype(int)
    with pytest.raises(ValueError, match="is_rest"):
        make_channels(**kw)


def test_1d_values_raises():
    kw = _base_kwargs()
    kw["P_values"] = np.ones(T, dtype=float)  # (T,) instead of (T, 1)
    with pytest.raises(ValueError, match="P_values"):
        make_channels(**kw)
