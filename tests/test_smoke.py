"""Import-level smoke tests. Keeps the package from silently breaking.

Run: pytest tests/test_smoke.py
"""
import run_modeling


def test_package_imports():
    assert len(run_modeling.__all__) > 0


def test_constants_present():
    from run_modeling import (
        WORLD_RECORD_SPEEDS,
        RIEGEL_DISTANCES_M,
        DECONFOUNDING_NORMS,
        DECONFOUNDING_LOG_COVARIATES,
    )
    assert len(RIEGEL_DISTANCES_M) > 0
    assert len(DECONFOUNDING_NORMS) > 0
