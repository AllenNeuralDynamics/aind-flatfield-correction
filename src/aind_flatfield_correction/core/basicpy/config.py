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
# Baseline parameters
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
