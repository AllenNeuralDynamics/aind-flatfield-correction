"""
Tuning constants and compute-environment settings for BaSiC fitting.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import psutil

# Use ladmap instead of approximate, in practice it gave better results
# Baseline parameters
BASIC_CONFIG = {
    "get_darkfield": False,
    "sort_intensity": True,
    "resize_mode": "skimage_dask",
    "fitting_mode": "ladmap",
    "max_reweight_iterations": 35,  # basicpy's default is only 10
    # Provable no-ops while get_darkfield=False (every use sits inside
    # `if self.get_darkfield:` in both solvers).  Pinned to the original
    # hand-tuned values for provenance; NOT searched.
    "smoothness_darkfield": 20,
    "sparse_cost_darkfield": 0.01,
}

# Known-good hand-tuned parameter.  Seeds the search, anchors the
# histogram window, and is the fallback whenever nothing beats it.
MANUAL_PARAMS = {"smoothness_flatfield": 1.0}

# The search is 1-D: smoothness_flatfield is the only effective dimension
# while get_darkfield=False.  The baseline value is always included.
SEARCH_GRID = sorted(
    set(
        [round(float(v), 6) for v in np.logspace(-2, 1, 10)]
        + [MANUAL_PARAMS["smoothness_flatfield"]]
    )
)

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
