"""run_modeling — per-athlete cardiorespiratory state estimation.

Organized around the DAG in ``docs/theoretical_framework.md``:

  Observation model  p(X | Z, P, E)
  State transitions  f (workout), g (rest, 10-day bound)
  Prediction         = state forward + observation at reference conditions

Module implementation follows ``docs/architecture_map.md``; at this stage
only shared constants are exposed.
"""

from ._constants import (
    WORLD_RECORD_SPEEDS,
    RIEGEL_DISTANCES_M,
    DEFAULT_NORMS,
    DECONFOUNDING_NORMS,
    DECONFOUNDING_LOG_COVARIATES,
)

__all__ = [
    "WORLD_RECORD_SPEEDS",
    "RIEGEL_DISTANCES_M",
    "DEFAULT_NORMS",
    "DECONFOUNDING_NORMS",
    "DECONFOUNDING_LOG_COVARIATES",
]
