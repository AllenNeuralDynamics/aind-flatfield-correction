"""
Provenance and run-log capture for flatfield estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from aind_flatfield_correction.core.basicpy.config import (
    CONFIRM_SEED,
    SUBSAMPLE_SEED,
    get_cpu_limit,
    worker_pool_size,
)

logger = logging.getLogger(__name__)

PACKAGE_NAME = "aind-flatfield-correction"

# The dependencies a fitted flatfield actually depends on: the solver,
# its linear-algebra backend and the array/imaging libraries. A pinned
# version changing is enough to change the result, so a run has to be
# able to say which ones it used.
TRACKED_DEPENDENCIES = (
    "BaSiCPy",
    "jax",
    "jaxlib",
    "numpy",
    "scikit-image",
    "dask",
    "zarr",
    "tifffile",
    "aind-large-scale-prediction",
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def utc_now() -> datetime:
    """
    Return the current time as an aware UTC timestamp.

    Returns
    -------
    datetime.datetime
        The current UTC time.
    """
    return datetime.now(timezone.utc)


def _package_version(name: str) -> str | None:
    """
    Look up an installed distribution's version.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str or None
        The version, or None when the distribution is not installed.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """
    Return the commit the package is running from, if it is a checkout.

    Best effort by design: an installed wheel or a Code Ocean capsule
    without git history simply has no commit to report, and that must
    not fail a run.

    Returns
    -------
    str or None
        The full commit hash, or None when it cannot be determined.
    """
    repo = Path(__file__).resolve().parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def sha256_file(path: str | Path) -> str | None:
    """
    Hash a file so a changed input is detectable later.

    Parameters
    ----------
    path : str or Path
        File to hash.

    Returns
    -------
    str or None
        Hex digest, or None when the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        logger.warning("  could not hash %s: %s", path, error)
        return None
    return digest.hexdigest()


def gather_provenance(
    args: argparse.Namespace,
    started: datetime,
    finished: datetime,
) -> dict[str, Any]:
    """
    Collect everything needed to reproduce a run.

    ``args`` is recorded in full rather than key by key, so a flag added
    to the parser later is captured without anyone having to remember to
    extend this function.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    started : datetime.datetime
        When the run began.
    finished : datetime.datetime
        When the run finished.

    Returns
    -------
    dict
        Code, environment, timing and argument provenance.
    """
    n_workers, threads = worker_pool_size()
    record: dict[str, Any] = {
        "command": list(sys.argv),
        "package_version": _package_version(PACKAGE_NAME),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dependencies": {
            name: _package_version(name) for name in TRACKED_DEPENDENCIES
        },
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "cpu_limit": get_cpu_limit(),
        "workers": {"processes": n_workers, "threads_each": threads},
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "seeds": {
            "stratified_subsample": SUBSAMPLE_SEED,
            "confirmation_subsample": CONFIRM_SEED,
        },
        "args": dict(vars(args)),
    }
    if getattr(args, "darkfield_image", None):
        record["darkfield_image_sha256"] = sha256_file(args.darkfield_image)
    return record


def attach_run_log(path: str | Path) -> logging.Handler:
    """
    Also write the log to ``path``, changing nothing else.

    Parameters
    ----------
    path : str or Path
        Log file to write; its parent must already exist.

    Returns
    -------
    logging.Handler
        The attached handler, to be passed to :func:`detach_run_log`.
    """
    root = logging.getLogger()
    handler = logging.FileHandler(path, mode="w")
    inherited = next(
        (
            existing.formatter
            for existing in root.handlers
            if existing.formatter is not None
        ),
        None,
    )
    handler.setFormatter(inherited or logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler) -> None:
    """
    Stop writing to a run log and close it.

    Called from a ``finally`` block so that a run which raises still
    leaves a complete, flushed log behind.

    Parameters
    ----------
    handler : logging.Handler
        The handler returned by :func:`attach_run_log`.

    Returns
    -------
    None
    """
    logging.getLogger().removeHandler(handler)
    handler.close()
