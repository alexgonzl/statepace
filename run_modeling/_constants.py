"""Shared constants used across activity_models submodules."""
import numpy as np
import pandas as pd

WORLD_RECORD_SPEEDS = pd.DataFrame(dict(
    distance=[1500, 1609, 3000, 5000, 10000, 20000, 21097, 40000, 42195],
    time=[206, 223, 438, 755, 1571, 3380, 3401, 6863, 7235],
))
WORLD_RECORD_SPEEDS['speed'] = WORLD_RECORD_SPEEDS['distance'] / WORLD_RECORD_SPEEDS['time']
WORLD_RECORD_SPEEDS['names'] = ["1.5k", "Mile", "3k", "5k", "10k", "20k", "Half Marathon", "40k", "Marathon"]
WORLD_RECORD_SPEEDS['num'] = np.arange(len(WORLD_RECORD_SPEEDS))
WORLD_RECORD_SPEEDS.set_index('distance', inplace=True)

RIEGEL_DISTANCES_M = [1500, 3000, 5000, 10000, 20000, 40000]

# ---------------------------------------------------------------------------
# Signal normalization — canonical source (re-exported by signal_utilities.py)
# ---------------------------------------------------------------------------
DEFAULT_NORMS = {
    'HR': 200.0, 'speed': 8.0, 'speed_h': 8.0, 'speed_v': 4.0, 'ca': 120.0,
    't': 3600.0, 'dist': 20000.0, 'asc': 200.0, 'dsc': 200.0,
}

# ---------------------------------------------------------------------------
# Deconfounding normalization — domain norms for covariate scaling
# ---------------------------------------------------------------------------
# (center, scale) pairs.  Coefficient interpretation:
#   "Δ S_hat (m/s) per 1 scale-unit increase from center"
# Preserves cross-subject comparability (no per-sample StandardScaler).
DECONFOUNDING_NORMS: dict[str, tuple[float, float]] = {
    'temperature':      (15.0, 10.0),   # °C; thermoneutral zone, 10°C range
    'humidity':         (50.0, 25.0),    # %; mid-range, ±25% covers most conditions
    'time_of_day':      (10.0, 6.0),     # hours; morning baseline, 6h span
    'elevation_gain':   (0.0, 200.0),    # meters; flat baseline, 200m = DEFAULT_NORMS['asc']
    'hr_variance':      (0.0, 50.0),     # bpm²; zero baseline, 50 ~ moderate variability
    'cadence_variance': (0.0, 10.0),     # spm²; zero baseline
    # covariate_df column names — prediction layer + WOLS alignment
    'wet_bulb_temp':      (15.0, 10.0),  # derived from temperature norm
    'elev_gain':          ( 0.0, 200.0), # alias of elevation_gain
    'elevation':          ( 0.0,   1.0), # sea-level centered; effectively unscaled
    'tod_sin':            ( 0.0,   1.0), # already [-1, 1]
    'tod_cos':            ( 0.0,   1.0), # already [-1, 1]
    'acwr_low':           ( 0.0,   1.0), # binary indicator
    'acwr_high':          ( 0.0,   1.0), # binary indicator
    'log_distance_m':     ( 0.0,   1.0), # already log-scaled
    'log_effort_start_s': ( 0.0,   1.0), # already log-scaled
    # cardiac cost covariate scales
    'net_grade':          ( 0.0,   0.1),           # slope fraction; 0.1 = 10% grade
    'elevation_m':        ( 0.0, 1000.0),           # m; coefficient = per km elevation
    'log_t0':             ( 0.0, np.log(600)),      # log-seconds; scale = 10 min into session
    # step load: 90 steps/min × 60 min × 3 m/s = 16200 (1hr at moderate pace)
    'chronic_step_load':  ( 0.0, 16200.0),
    'acute_step_load':    ( 0.0, 16200.0),
    # hr load: zone-2 weight (2) × 3600 s = 7200 (1hr at zone 2)
    'chronic_hr_load':    ( 0.0, 7200.0),
    'acute_hr_load':      ( 0.0, 7200.0),
}

# Covariates that use log(x / reference) instead of (x - center) / scale.
# Motivated by Riegel power-law: speed ~ distance^alpha => log-linear.
# Coefficient interpretation: "Δ S_hat (m/s) per unit of log(distance/ref)"
DECONFOUNDING_LOG_COVARIATES: dict[str, float] = {
    'total_distance': 5000.0,  # reference 5km session
}
