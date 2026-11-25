"""
Tuning constants and compute-environment settings for BaSiC fitting.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)

SMOOTHNESS_KEY = "smoothness_flatfield"

# Use ladmap instead of approximate, in practice it gave better results
# Baseline parameters.
#
# smoothness_darkfield and sparse_cost_darkfield do nothing while
# get_darkfield is False.  If get_darkfield is turned on, the value 20 makes
# the LADMAP solver shrink BaSiC's per-pixel darkfield to exactly zero, so
# only one constant can be learned.  Do not rely on BaSiC's own darkfield
# in any case: on dim HCR and on SmartSPIM channels it returned 0-6 counts
# against a measured camera dark of 90-104, with these settings and with
# basicpy's defaults.  Pass a measured darkfield instead.
BASIC_CONFIG = {
    "get_darkfield": False,
    "sort_intensity": True,
    "resize_mode": "skimage_dask",
    "fitting_mode": "ladmap",
    "max_reweight_iterations": 35,
    "smoothness_darkfield": 20,
    "sparse_cost_darkfield": 0.01,
    SMOOTHNESS_KEY: 1.0,
}

# Objective dynamic range on this data is only ~5e-3 nats, so a candidate
# must show a real win before it displaces the baseline.
SELECT_TOL = 1e-4

# Flatfield plausibility thresholds.  basicpy normalizes the flatfield
# mean to exactly 1.0, so std and span are scale-free and comparable
# across channels.  Known-good manual flatfields measured std
# 0.082-0.112 / span 0.369-0.482; broken autotuned ones were std
# 0.011-0.023 / span 0.053-0.103.  The ceiling is loose enough that a
# valid fit on pedestal-subtracted data (span ~0.7) does not trip it.
FF_STD_FLOOR = 0.030
FF_SPAN_FLOOR = 0.100
FF_STD_CEILING = 0.50
FF_REL_FLOOR = 0.60

# Ceiling on the largest boost of the correction, p99.9 / min of the
# flatfield: how many times dividing by the flatfield brightens its darkest
# part relative to its bright part.  Accepted flatfields of HCR and
# SmartSPIM data measured 1.4-2.9; flatfields that learned tissue measured
# 17-358 (SmartSPIM 561) while some still passed the std ceiling.
FF_BOOST_CEILING = 5.0

# Pre-fit checks (written to the sidecar, never fatal).
#
# Dark error over signal: the flatfield error is about the dark error
# divided by the signal.  It measured 25-150% on dim HCR channels, whose
# scalar-dark flats were worse than no flatfield, and 0.6-1.4% on
# SmartSPIM channels.
DARK_ERROR_WARN = 0.10
# Signal (median plane mean above the dark) below which a scalar dark is
# likely to leave the camera pattern (0.7-3 counts std in the data
# measured) in the flatfield.  Dim HCR channels: 0.3-9 counts; SmartSPIM
# channels: 106-162 counts.
DIM_SIGNAL_WARN = 30.0
# Share of the fit objective held by the brightest planes
# (``tiles.plane_concentration``) above which the signal is concentrated:
# healthy channels 0.14-0.33, SmartSPIM 561 0.55-0.63.
CONCENTRATION_WARN = 0.45
# Second sign of concentrated signal, for stacks in which many planes are
# affected, so that the brightest few hold only a modest share of the
# objective: the share of all planes that are bright and concentrated
# (``tiles.concentrated_planes``).  SmartSPIM 561: about 20% per laser
# side; healthy channels 0-6%; a synthetic stack with a bright region in
# 30% of its images: 28%, while its top-plane share was only 0.15.
CONCENTRATED_PLANES_WARN = 0.10

# Plane filter (--drop-concentrated-planes).  A plane is dropped when it is
# bright, its mean above BRIGHT_RATIO x the median plane mean, AND
# concentrated, the 99th percentile of its background map above
# LOCALIZED_MAX x the map's median.  On SmartSPIM 561 this dropped about
# 20% of the planes per laser side and removed the tissue from the
# flatfield; on healthy channels it dropped 0-6% and left the flatfield
# unchanged.  Dropping every bright plane instead removed the best planes
# of a channel whose bright tissue covers the whole view.
BRIGHT_RATIO = 3.0
LOCALIZED_MAX = 3.0
# Number of blocks on the short side of the background map, so that the
# same physical block size is used at every pyramid level.
BG_GRID_BLOCKS = 40

# Share of the fitted planes, the dimmest by mean, whose mean is subtracted
# for the signal CV.  The darkfield cancels in that difference.
SIGNAL_LOW_FRACTION = 0.10

# Darkfield from the darkest tiles (``darkmap``).  The darkest plane of the
# DARKMAP_N_TILES tiles whose darkest plane is darkest: on an HCR channel
# whose dark planes carried scattered light, this took the flatfield's
# signal CV from 3.2% (6 fixed tiles) to 1.35%, and 6, 12 or 24 tiles gave
# the same result within the fit noise (1.31-1.42%).
DARKMAP_N_TILES = 12
# Spread (95th - 5th percentile) of the darkest-plane level across tiles
# above which the dark planes contain light: a camera dark is the same in
# every tile.  HCR channels with a camera-only dark: 0.7-1.4 counts; a
# channel whose dark planes carried scattered light: 8.3 counts.
LIGHT_SPREAD_WARN = 3.0
# Share of the tiles whose darkest plane is below the darkfield's mean,
# above which the darkfield is too high for many tiles and removes real
# signal there.  6 fixed tiles: 63%; the 12 darkest tiles: 3%.
OVER_SUBTRACTED_WARN = 0.10
# Pixel step used to measure the mean of every plane when ranking planes.
LEVEL_SUBSAMPLE = 4

# Percentile of the accepted flatfield that is rescaled to 1.0 before it
# is saved.  Dividing by basicpy's mean-1 flatfield dims every pixel
# brighter than the mean -- the tile centre, by 9-24% on SmartSPIM and
# HCR data -- and so compresses the signal.  With the 99.9th percentile
# at 1.0, only 0.1% of pixels keep a gain below 1, by about 1% or less
# on smooth flats; everywhere else the correction only brightens.  The
# record written by ``fit.scale_flatfield`` reports the actual worst
# case, which a flat with a long upper tail raises.  The guard runs
# before the rescaling, on the mean-1 field its thresholds assume.
FLAT_SCALE_PERCENTILE = 99.9

# Seeds for the two subsamples the search draws.  Named so that a run
# can record them rather than duplicating the literals.
SUBSAMPLE_SEED = 42
CONFIRM_SEED = 7

# Fractional Z position of the plane used for the inspection figures.
Z_FRACTION = 0.3


def baseline_params(basic_config: dict[str, Any]) -> dict[str, float]:
    """
    Extract the incumbent parameters a configuration carries.

    The search compares candidates against whatever
    ``smoothness_flatfield`` the configuration was given, so a run that
    supplies its own hand-tuned value gets that value as the incumbent
    rather than the built-in default.

    Parameters
    ----------
    basic_config : dict
        Resolved BaSiC solver configuration.

    Returns
    -------
    dict
        The searched parameters, at their incumbent values.
    """
    return {
        SMOOTHNESS_KEY: float(
            basic_config.get(SMOOTHNESS_KEY, BASIC_CONFIG[SMOOTHNESS_KEY])
        )
    }


def search_grid(baseline: float) -> list[float]:
    """
    Build the candidate grid, always including the incumbent.

    The search is 1-D: ``smoothness_flatfield`` is the only effective
    dimension while ``get_darkfield`` is False.  The incumbent is always
    a candidate so that its score is directly comparable with the rest.

    Parameters
    ----------
    baseline : float
        The incumbent ``smoothness_flatfield``.

    Returns
    -------
    list of float
        Sorted, unique candidate values.
    """
    decades = [round(float(value), 6) for value in np.logspace(-2, 1, 10)]
    return sorted(set(decades + [round(float(baseline), 6)]))


def load_basic_config(source: str | None = None) -> dict[str, Any]:
    """
    Resolve the BaSiC solver configuration for one run.

    The built-in :data:`BASIC_CONFIG` is the known-good baseline, so a
    caller-supplied configuration is MERGED over it rather than
    replacing it: only the keys being changed have to be given, and
    anything omitted keeps its vetted value.

    Parameters
    ----------
    source : str or None, optional
        A path to a JSON file, or an inline JSON object (anything
        starting with ``{`` or ``[``). None keeps the built-in
        defaults.

    Returns
    -------
    dict
        The configuration to hand to BaSiC.

    Raises
    ------
    FileNotFoundError
        If ``source`` looks like a path and no such file exists.
    ValueError
        If the JSON does not describe an object.
    json.JSONDecodeError
        If the JSON cannot be parsed.
    """
    if not source:
        return dict(BASIC_CONFIG)

    text = source.strip()
    # Anything opening with JSON punctuation is parsed as JSON, so a
    # list reaches the object check below rather than being mistaken
    # for a filename.
    if text.startswith(("{", "[")):
        overrides = json.loads(text)
    else:
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(f"BaSiC config file not found: {source}")
        overrides = json.loads(path.read_text())

    if not isinstance(overrides, dict):
        raise ValueError(
            "BaSiC config must be a JSON object, got "
            f"{type(overrides).__name__}"
        )

    config = {**BASIC_CONFIG, **overrides}
    if overrides:
        logger.info(
            "  BaSiC config overridden: %s",
            {key: config[key] for key in sorted(overrides)},
        )
    return config


def limit_worker_threads(threads_per_worker: int) -> None:
    """
    Pin a worker process to CPU JAX and a bounded BLAS thread count.

    Must run before JAX or numpy do any work: 8 processes x 16 default
    BLAS threads on 16 cores thrash and cost 2-3x in contention.

    Parameters
    ----------
    threads_per_worker : int
        Threads each worker process may use.

    Returns
    -------
    None
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(max(1, threads_per_worker)))


def get_cpu_limit() -> int:
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        # int(): the env var is a string, and worker_pool_size compares
        # it against an int.
        return int(co_cpus)
    if aws_batch_job_id:
        return 1

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError:
        # Not running under a cgroup v1 CPU quota.
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return (
        psutil.cpu_count(logical=False)
        if container_cpus < 1
        else container_cpus
    )


def worker_pool_size(n_workers: Optional[int] = 8) -> tuple[int, int]:
    """
    Choose the worker count and the per-worker thread count.

    Parameters
    ----------
    n_workers : int, optional
        Ceiling on the number of worker processes, by default 8. The
        available CPU count is used when it is lower.

    Returns
    -------
    tuple of (int, int)
        Number of worker processes and threads per worker.
    """
    n_cpu = max(1, int(get_cpu_limit() or 1))
    n_workers = max(1, min(n_cpu, n_workers or 1))
    return n_workers, max(1, n_cpu // n_workers)
