"""
Flatfield fitting and plausibility checks.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from basicpy import BaSiC

from aind_flatfield_correction.core.basicpy.config import (
    FF_REL_FLOOR,
    FF_SPAN_FLOOR,
    FF_STD_CEILING,
    FF_STD_FLOOR,
    limit_worker_threads,
    worker_pool_size,
)

logger = logging.getLogger(__name__)


def validate_basic_config(config: dict[str, Any]) -> None:
    """
    Construct a BaSiC with ``config`` so a bad key fails immediately.

    Worth doing before anything else: the configuration is only used
    after every tile has been streamed in, so an unknown or ill-typed
    key would otherwise surface as a solver error half an hour into a
    run.

    Parameters
    ----------
    config : dict
        BaSiC keyword arguments to check.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Whatever BaSiC raises for the offending key, after logging the
        configuration that was rejected.
    """
    try:
        BaSiC(**config)
    except Exception:
        logger.error("BaSiC rejected the configuration %s", config)
        raise


def _fit_one_z(
    shm_name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    zi: int,
    base_config: dict[str, Any],
    threads_per_worker: int = 2,
) -> tuple[int, np.ndarray | None, str | None]:
    """
    Worker for the per-Z method: fit one Z index across all tiles.

    The shared array is ``(n_planes, n_tiles, H, W)``; this fits slice
    ``zi`` -- that is, ``n_tiles`` images of the same Z plane -- and
    returns just the ``(H, W)`` flatfield.

    Parameters
    ----------
    shm_name : str
        Name of the shared-memory block holding the stacks.
    shape : tuple of int
        Shape of the shared array.
    dtype : np.dtype
        Dtype of the shared array.
    zi : int
        Z index to fit.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    threads_per_worker : int, optional
        BLAS thread cap, by default 2.

    Returns
    -------
    tuple of (int, np.ndarray or None, str or None)
        The Z index, the fitted flatfield (None on failure), and the
        traceback (None on success).
    """
    limit_worker_threads(threads_per_worker)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        stacks = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        basic = BaSiC(**base_config)
        basic.fit(np.ascontiguousarray(stacks[zi]))
        flat = np.asarray(basic.flatfield, dtype=np.float32)
        return int(zi), flat, None
    except Exception:  # noqa: BLE001 - reported to the parent
        import traceback

        return int(zi), None, traceback.format_exc()
    finally:
        shm.close()


def _stack_planes_by_z(
    all_slices: np.ndarray,
    tile_z_offsets: dict[int, tuple[int, int]],
    n_planes: int,
) -> np.ndarray:
    """
    Regroup a concatenated stack into ``(n_planes, n_tiles, H, W)``.

    Each tile's planes are contiguous per ``tile_z_offsets``, so plane
    ``k`` of every tile is gathered by taking index ``start + k`` from
    each span.

    Parameters
    ----------
    all_slices : np.ndarray
        Concatenated ``(n_tiles * n_planes, H, W)`` stack.
    tile_z_offsets : dict
        Per-tile ``(start, end)`` spans into ``all_slices``.
    n_planes : int
        Planes taken per tile.

    Returns
    -------
    np.ndarray
        ``(n_planes, n_tiles, H, W)`` float32 array.
    """
    spans = [tile_z_offsets[key] for key in sorted(tile_z_offsets)]
    height, width = all_slices.shape[1:]
    stacks = np.zeros((n_planes, len(spans), height, width), dtype=np.float32)
    for tile_i, (start, end) in enumerate(spans):
        available = end - start
        for plane in range(n_planes):
            stacks[plane, tile_i] = all_slices[
                start + min(plane, available - 1)
            ]
    return stacks


def _collect_per_z_flatfields(
    stacks: np.ndarray,
    n_planes: int,
    base_config: dict[str, Any],
) -> dict[int, np.ndarray]:
    """
    Fit every Z plane in parallel and collect the successful fits.

    Parameters
    ----------
    stacks : np.ndarray
        ``(n_planes, n_tiles, H, W)`` array; consumed via shared memory.
    n_planes : int
        Number of independent fits to run.
    base_config : dict
        BaSiC keyword arguments, used verbatim.

    Returns
    -------
    dict
        ``{z_index: flatfield}`` for the fits that succeeded.
    """
    n_workers, threads = worker_pool_size()
    flats: dict[int, np.ndarray] = {}
    failures = 0
    shm = shared_memory.SharedMemory(create=True, size=stacks.nbytes)
    try:
        view = np.ndarray(stacks.shape, dtype=stacks.dtype, buffer=shm.buf)
        view[:] = stacks
        shape, dtype = stacks.shape, stacks.dtype
        del view
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_ctx
        ) as pool:
            futures = [
                pool.submit(
                    _fit_one_z,
                    shm.name,
                    shape,
                    dtype,
                    zi,
                    base_config,
                    threads,
                )
                for zi in range(n_planes)
            ]
            done = 0
            for future in as_completed(futures):
                zi, flat, err = future.result()
                done += 1
                if err:
                    failures += 1
                    logger.error("    z-fit %d FAILED:\n%s", zi, err)
                else:
                    flats[zi] = flat
                if done % 5 == 0 or done == n_planes:
                    logger.info(
                        "    %d/%d z-fits done (%d failed)",
                        done,
                        n_planes,
                        failures,
                    )
    finally:
        shm.close()
        shm.unlink()
    return flats


def estimate_per_z_median(
    all_slices: np.ndarray,
    tile_z_offsets: dict[int, tuple[int, int]],
    n_planes: int,
    base_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fit one BaSiC per Z index and combine the results by median.

    Mirrors the production capsule: for each of the ``n_planes`` Z
    positions, fit BaSiC on that plane across ALL tiles, then take the
    pixelwise median of the per-plane flatfields.  Each fit is small,
    they are independent, and the median rejects planes whose fit went
    wrong.

    Parameters
    ----------
    all_slices : np.ndarray
        Concatenated ``(n_tiles * n_planes, H, W)`` fitting stack.
    tile_z_offsets : dict
        Per-tile ``(start, end)`` spans into ``all_slices``.
    n_planes : int
        Planes taken per tile, i.e. the number of independent fits.
    base_config : dict
        BaSiC keyword arguments, used verbatim.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, int)
        The median flatfield, a zero darkfield of the same shape, and
        the number of per-Z fits that succeeded.

    Raises
    ------
    RuntimeError
        If every per-Z fit failed.
    """
    stacks = _stack_planes_by_z(all_slices, tile_z_offsets, n_planes)
    logger.info(
        "  per-z-median: %d independent fits of %d images each",
        n_planes,
        stacks.shape[1],
    )
    flats = _collect_per_z_flatfields(stacks, n_planes, base_config)
    del stacks
    if not flats:
        raise RuntimeError("every per-Z fit failed - cannot build a flatfield")

    stack = np.stack([flats[k] for k in sorted(flats)], axis=0)
    flatfield = np.median(stack, axis=0).astype(np.float32)
    logger.info(
        "  Combined %d/%d per-Z flatfields by median (mean "
        "plane-to-plane std=%.4f)",
        stack.shape[0],
        n_planes,
        float(stack.std(axis=0).mean()),
    )
    return flatfield, np.zeros_like(flatfield), int(stack.shape[0])


def estimate_joint(
    fit_slices: np.ndarray, fit_config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, None]:
    """
    Fit a single BaSiC model over every gathered plane at once.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.
    fit_config : dict
        Full BaSiC keyword arguments, including the chosen smoothness.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, None)
        The flatfield, the darkfield reported by BaSiC, and None (this
        method runs no per-Z fits to count).
    """
    logger.info("  Fitting BaSiC on %d images", fit_slices.shape[0])
    basic = BaSiC(**fit_config)
    basic.fit(images=fit_slices)
    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield = np.asarray(basic.darkfield, dtype=np.float32)
    return flatfield, darkfield, None


def check_flatfield_plausibility(
    flatfield: np.ndarray, baseline_std: float | None = None
) -> tuple[bool, dict[str, float], list[str]]:
    """
    Sanity-check a fitted flatfield.

    This is the check that catches an inverted objective: it fires when
    the flatfield is so flat that the correction is a no-op.

    These are some heuristics based on experiments performed with basicpy.
    Please, double check with your data.

    Parameters
    ----------
    flatfield : np.ndarray
        Fitted flatfield.
    baseline_std : float or None, optional
        Standard deviation of the baseline fit, for the relative floor.

    Returns
    -------
    tuple of (bool, dict, list of str)
        Whether the flatfield passed, its statistics, and one reason
        string per failed check.
    """
    flat = np.asarray(flatfield, dtype=np.float64)
    stats = {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "span": float(np.ptp(flat)),
        "std": float(flat.std()),
        "mean": float(flat.mean()),
    }
    reasons = []
    if not np.isfinite(flat).all():
        reasons.append("contains non-finite values")
    if stats["min"] <= 0:
        reasons.append(f"min {stats['min']:.4f} <= 0 (division would blow up)")
    if stats["std"] < FF_STD_FLOOR:
        reasons.append(
            f"std {stats['std']:.4f} < floor {FF_STD_FLOOR} "
            "(correction is a no-op)"
        )
    if stats["span"] < FF_SPAN_FLOOR:
        reasons.append(
            f"span {stats['span']:.4f} < floor {FF_SPAN_FLOOR} "
            "(correction is a no-op)"
        )
    if stats["std"] > FF_STD_CEILING:
        reasons.append(
            f"std {stats['std']:.4f} > ceiling {FF_STD_CEILING} "
            "(over-fit / runaway)"
        )
    if baseline_std and stats["std"] < FF_REL_FLOOR * baseline_std:
        reasons.append(
            f"std {stats['std']:.4f} < {FF_REL_FLOOR:.0%} of baseline "
            f"std {baseline_std:.4f}"
        )
    return (not reasons), stats, reasons


def report_flatfield(
    flatfield: np.ndarray,
    baseline_std: float | None = None,
    label: str = "",
) -> tuple[bool, dict[str, float], list[str]]:
    """
    Run the flatfield plausibility guard and log a banner
    when it fails.

    Parameters
    ----------
    flatfield : np.ndarray
        Fitted flatfield.
    baseline_std : float or None, optional
        Standard deviation of the baseline fit.
    label : str, optional
        Extra tag for the log line.

    Returns
    -------
    tuple of (bool, dict, list of str)
        Whether the flatfield passed, its statistics, and the reasons.
    """
    ok, stats, reasons = check_flatfield_plausibility(flatfield, baseline_std)
    logger.info(
        "  Flatfield%s: min=%.4f max=%.4f span=%.4f std=%.4f",
        f" {label}" if label else "",
        stats["min"],
        stats["max"],
        stats["span"],
        stats["std"],
    )
    if not ok:
        bar = "  " + "!" * 68
        logger.warning(bar)
        logger.warning(
            "  SUSPICIOUS FLATFIELD! Data might be wrongly corrected! "
        )
        for reason in reasons:
            logger.warning("  !!!   - %s", reason)
        logger.warning(bar)
    return ok, stats, reasons
