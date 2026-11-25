"""
Estimate flatfields using BaSiC.

One run fits one multiplicative flatfield from a bounded number of Z
planes taken from every tile under ``base_path``.  The sensor camera
offset is removed before fitting.

The saved flatfield is rescaled so that its 99.9th percentile, rather
than its mean, is 1.0: dividing by basicpy's mean-1 field would dim the
bright centre of every tile.

This module holds the estimation pipeline; the command-line entry point
lives in :mod:`aind_flatfield_correction.core.basicpy.cli`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from matplotlib.image import imsave

from aind_flatfield_correction.core.basicpy.config import (
    CONCENTRATED_PLANES_WARN,
    CONCENTRATION_WARN,
    DARK_ERROR_WARN,
    DIM_SIGNAL_WARN,
    baseline_params,
)
from aind_flatfield_correction.core.basicpy.figures import (
    save_validation_figures,
)
from aind_flatfield_correction.core.basicpy.fit import (
    estimate_joint,
    estimate_per_z_median,
    report_flatfield,
    scale_flatfield,
)
from aind_flatfield_correction.core.basicpy.search import (
    confirm_against_baseline,
    parallel_autotune,
)
from aind_flatfield_correction.core.basicpy.tiles import (
    FULL_RESOLUTION_LEVEL,
    FitStack,
    concentrated_planes,
    drop_concentrated_planes,
    load_fit_stack,
    match_darkfield,
    plane_concentration,
    plane_scores,
    probe_plane_shape,
    resize_plane,
    subtract_pedestal,
)

logger = logging.getLogger(__name__)

METADATA_DIR = "metadata"

RUN_LOG_NAME = "estimation.log"


class FitResult(NamedTuple):
    """
    Outcome of the final fit, after the plausibility guard.
    Attributes
    ----------
    flatfield : np.ndarray
        The estimated multiplicative flatfield.
    darkfield : np.ndarray
        The estimated additive darkfield (camera offset).
    n_z_fits_ok : int or None
        Number of successful per-Z fits (None for the joint method).
    params : dict[str, float]
        The parameters used for the final fit.
    ok : bool
        Whether the fit passed the plausibility check.
    stats : dict[str, float]
        Summary statistics of the fit.
    reasons : list[str]
        Reasons for any plausibility check failures.
    """

    flatfield: np.ndarray
    darkfield: np.ndarray
    n_z_fits_ok: int | None
    params: dict[str, float]
    ok: bool
    stats: dict[str, float]
    reasons: list[str]


def prefit_checks(
    fit_slices: np.ndarray,
    dark_plane: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Check the fitting stack for the conditions that broke flatfields.

    Two conditions were found to decide whether a flatfield is valid,
    and both can be measured before the fit.

    * **The darkfield matters.**  The flatfield error is about the
      darkfield error divided by the signal.  With a per-pixel darkfield,
      ``dark_error_ratio`` is how far it is from a scalar, relative to the
      signal: above :data:`DARK_ERROR_WARN` the darkfield decides the
      flatfield and must be accurate.  With a scalar darkfield the error
      cannot be measured, so a signal below :data:`DIM_SIGNAL_WARN` counts
      is flagged instead: the camera pattern a scalar leaves behind is then
      a large part of the signal.
    * **The signal is concentrated.**  When the brightest planes hold more
      than :data:`CONCENTRATION_WARN` of the fit objective, or when more
      than :data:`CONCENTRATED_PLANES_WARN` of all planes are bright and
      have one dominant bright region, the flatfield can learn the tissue
      (see :func:`tiles.drop_concentrated_planes`).  The second sign
      catches stacks in which so many planes are affected that the
      brightest few hold only a modest share of the objective.
      ``concentrated_signal`` records the verdict; with
      ``--drop-concentrated-planes`` it decides whether planes are dropped.

    The checks only warn and are recorded in the sidecar; nothing is
    refused.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted fitting stack.
    dark_plane : np.ndarray
        Darkfield plane subtracted from it.
    scores : dict
        Output of :func:`tiles.plane_scores` for ``fit_slices``.

    Returns
    -------
    dict
        The measured values and one message per warning.
    """
    signal = float(np.median(scores["mean"]))
    dark = np.asarray(dark_plane, dtype=np.float64)
    dark_error = float(np.std(dark - np.median(dark)))
    per_pixel = dark_error > 0
    concentration = plane_concentration(fit_slices)
    bright, both = concentrated_planes(scores)
    concentrated_fraction = float(both.mean()) if both.size else 0.0
    checks: dict[str, Any] = {
        "signal": signal,
        "per_pixel_darkfield": per_pixel,
        "dark_error_std": dark_error if per_pixel else None,
        "dark_error_ratio": (
            dark_error / signal if per_pixel and signal > 0 else None
        ),
        "plane_concentration": concentration,
        "bright_planes": int(bright.sum()),
        "concentrated_share_of_bright": (
            float(both.sum() / bright.sum()) if bright.any() else 0.0
        ),
        "concentrated_planes": int(both.sum()),
        "concentrated_fraction": concentrated_fraction,
        "concentrated_signal": bool(
            concentration["top_share"] > CONCENTRATION_WARN
            or concentrated_fraction > CONCENTRATED_PLANES_WARN
        ),
        "warnings": [],
    }
    warnings = checks["warnings"]
    if signal <= 0:
        warnings.append("no signal above the darkfield in the fit planes")
    elif per_pixel and checks["dark_error_ratio"] > DARK_ERROR_WARN:
        warnings.append(
            f"the darkfield matters: it differs from a scalar by "
            f"{checks['dark_error_ratio']:.0%} of the signal "
            f"({signal:.1f} counts), and the flatfield error is about the "
            "darkfield error over the signal, so the darkfield must be "
            "accurate"
        )
    elif not per_pixel and signal < DIM_SIGNAL_WARN:
        warnings.append(
            f"dim channel: the signal is {signal:.1f} counts above a scalar "
            "darkfield, which leaves the camera pattern in the flatfield; "
            "supply a measured per-pixel darkfield (recorded dark frames, or "
            "one built with core.basicpy.darkmap)"
        )
    if checks["concentrated_signal"]:
        warnings.append(
            f"concentrated signal: the brightest planes hold "
            f"{concentration['top_share']:.0%} of the fit objective and "
            f"{concentrated_fraction:.0%} of the planes are bright with one "
            "dominant region, so the flatfield can learn the tissue; "
            "--drop-concentrated-planes drops those planes"
        )
    for message in warnings:
        logger.warning("  Pre-fit check: %s", message)
    return checks


def select_parameters(
    fit_slices: np.ndarray,
    z_offsets: dict[int, tuple[int, int]],
    args: argparse.Namespace,
    basic_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any] | None]:
    """
    Choose ``smoothness_flatfield`` for the final fit.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.
    z_offsets : dict
        Per-tile spans into ``fit_slices``.
    args : argparse.Namespace
        Parsed command-line arguments.
    basic_config : dict
        BaSiC solver configuration every candidate is scored under.

    Returns
    -------
    tuple of (dict, dict, dict or None)
        The chosen parameters, the search report, and the confirmation
        report (None when no confirmation ran).
    """
    if args.skip_search:
        configured = baseline_params(basic_config)
        logger.info("  Using the configured params: %s", configured)
        return (
            configured,
            {"reason": "skip_search_flag", "chose_baseline": True},
            None,
        )

    # Parallel autotune, faster than basicpy serial search
    params, report = parallel_autotune(
        images=fit_slices,
        base_config=basic_config,
        tile_z_offsets=z_offsets,
        max_eval_slices=args.max_eval_slices,
        max_search_minutes=args.max_search_minutes,
    )
    # A winner found on a small subset must hold up nearer final scale.
    if not report.get("chose_baseline") and args.n_confirm > 0:
        params, confirm = confirm_against_baseline(
            fit_slices,
            basic_config,
            params,
            tile_z_offsets=z_offsets,
            n_confirm=args.n_confirm,
        )
        return params, report, confirm
    return params, report, None


def _run_fit(
    fit_slices: np.ndarray,
    stack: FitStack,
    method: str,
    fit_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """
    Dispatch to the requested estimation method.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted fitting stack.
    stack : FitStack
        Per-tile spans and plane count, needed by the per-Z method.
    method : str
        Either ``"fit"`` or ``"per-z-median"``.
    fit_config : dict
        Full BaSiC keyword arguments.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, int or None)
        The flatfield, the darkfield, and the number of successful per-Z
        fits (None for the joint method).
    """
    # We only have two methods, we can add more
    # per-z-median is an approximation, runs faster but could be
    # less accurate than the joint fit.
    if method == "per-z-median":
        return estimate_per_z_median(
            fit_slices, stack.z_offsets, stack.n_planes, fit_config
        )
    return estimate_joint(fit_slices, fit_config)


def fit_with_guard(
    fit_slices: np.ndarray,
    stack: FitStack,
    method: str,
    params: dict[str, float],
    baseline_std: float | None,
    basic_config: dict[str, Any],
) -> FitResult:
    """
    Run the final fit and fall back to manual params if it looks wrong.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted fitting stack.
    stack : FitStack
        Per-tile spans and plane count.
    method : str
        Either ``"fit"`` or ``"per-z-median"``.
    params : dict
        Parameters chosen by the search.
    baseline_std : float or None
        Standard deviation of the baseline fit, for the relative floor.
    basic_config : dict
        BaSiC solver configuration the chosen parameters overlay.

    Returns
    -------
    FitResult
        The flatfield actually accepted, the parameters that produced it,
        and the guard's verdict.
    """
    started = time.monotonic()
    flatfield, darkfield, n_z_ok = _run_fit(
        fit_slices, stack, method, {**basic_config, **params}
    )
    logger.info("  Fit took %.1f min", (time.monotonic() - started) / 60.0)
    ok, stats, reasons = report_flatfield(flatfield, baseline_std)

    configured = baseline_params(basic_config)
    if not ok and params != configured:
        logger.warning(
            "  Refitting with the configured params %s after the failed "
            "plausibility check",
            configured,
        )
        params = configured
        flatfield, darkfield, n_z_ok = _run_fit(
            fit_slices, stack, method, {**basic_config, **configured}
        )
        ok, stats, reasons = report_flatfield(
            flatfield, None, "(configured refit)"
        )
    if not ok:
        logger.warning(
            "  !! flatfield still suspicious after the fallback - "
            "recording it as suspicious"
        )
    return FitResult(flatfield, darkfield, n_z_ok, params, ok, stats, reasons)


def apply_flatfield_scaling(
    result: FitResult, args: argparse.Namespace
) -> tuple[FitResult, dict[str, Any]]:
    """
    Rescale the accepted flatfield so the correction stops dimming.

    Runs after the guard, which judged basicpy's mean-1 field on the
    thresholds calibrated for it; ``result.stats`` keep describing that
    field.  Everything downstream -- the saved arrays, the upsampled
    field and the figures -- gets the rescaled one.

    A flatfield that cannot be scaled is kept as fitted rather than
    failing the run after a long fit: the guard has already marked it
    suspicious, and the record says why it was left unscaled.

    Parameters
    ----------
    result : FitResult
        The fit accepted by the guard.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple of (FitResult, dict)
        The result carrying the flatfield to save, and the scaling
        record for the sidecar.
    """
    if not args.scale_flatfield:
        logger.info("  Flatfield kept at mean 1 (--no-scale-flatfield)")
        return result, {"applied": False, "reason": "--no-scale-flatfield"}
    try:
        scaled, record = scale_flatfield(
            result.flatfield, args.scale_percentile
        )
    except ValueError as error:
        logger.warning("  Flatfield left at mean 1: %s", error)
        return result, {"applied": False, "reason": str(error)}
    logger.info(
        "  Flatfield scaled so its %gth percentile is 1 (divided by %.4f): "
        "signal lowered on %.2f%% of pixels, by at most %.2f%%; "
        "boosted by at most %.1f%%",
        record["percentile"],
        record["scale"],
        100.0 * record["frac_pixels_reduced"],
        record["max_reduction_pct"],
        record["max_boost_pct"],
    )
    return result._replace(flatfield=scaled), {"applied": True, **record}


def write_tile_manifest(
    label: str, output_folder: Path, stack: FitStack
) -> str:
    """
    Write one record per tile that contributed to the fit.

    A separate file rather than a sidecar key: it is one entry per tile,
    which for a full channel would bury the parts of the sidecar a
    person actually reads.

    Parameters
    ----------
    label : str
        Name used for the output file.
    output_folder : Path
        Run's output folder; the metadata subfolder is created here.
    stack : FitStack
        The loaded stack, carrying its per-tile manifest.

    Returns
    -------
    str
        The manifest's path relative to ``output_folder``.
    """
    folder = output_folder / METADATA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"{METADATA_DIR}/tiles_{label}.json"
    records = [record._asdict() for record in stack.manifest]
    with open(output_folder / relative, "w") as handle:
        json.dump(records, handle, indent=2, default=str)
    logger.info("  Tile manifest: %d tiles -> %s", len(records), relative)
    return relative


def summarize_stack(
    stack: FitStack, n_fit_images: int, pedestal: dict[str, float]
) -> dict[str, Any]:
    """
    Summarize the stack the fit was built from.

    The median is taken from the pedestal report rather than recomputed:
    ``subtract_pedestal`` has already paid for it, and a second pass
    over a stack this size costs seconds and a large temporary.

    Parameters
    ----------
    stack : FitStack
        The loaded stack.
    n_fit_images : int
        Planes actually handed to the solver.
    pedestal : dict
        The report from ``subtract_pedestal``.

    Returns
    -------
    dict
        Extent, plane counts, intensity range and how many tiles were
        zero-padded into the stack.
    """
    return {
        "height": int(stack.height),
        "width": int(stack.width),
        "planes_per_tile": int(stack.n_planes),
        "n_images": int(n_fit_images),
        "min": float(stack.slices.min()),
        "max": float(stack.slices.max()),
        "median": pedestal["median_raw"],
        "n_padded_tiles": sum(1 for record in stack.manifest if record.padded),
    }


def upsample_to_full_resolution(
    label: str,
    output_folder: Path,
    flatfield: np.ndarray,
    dark: np.ndarray | float,
    tile_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Resample both fields up to the full-resolution pyramid level.

    The fit runs on a downsampled level for speed, but the correction is
    applied to full-resolution voxels, so both the flatfield and the
    darkfield have to be resampled to that extent.

    The darkfield is resampled from the original image (or the scalar
    camera offset), not from the plane that was matched to the estimation
    level.

    Parameters
    ----------
    label : str
        Name used for the output files.
    output_folder : Path
        Folder to write into.
    flatfield : np.ndarray
        The fitted flatfield, at the estimation level, already rescaled
        when scaling is on: the same factor then holds at both levels.
    dark : np.ndarray or float
        The darkfield as supplied: an image at its own resolution, or a
        scalar pedestal.
    tile_name : str
        Tile whose full-resolution extent stands for the channel's.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        The destination level, the shape written, the scale factor and
        the paths of both fields.
    """
    shape = probe_plane_shape(args.base_path, tile_name, FULL_RESOLUTION_LEVEL)
    suffix = f"_level{FULL_RESOLUTION_LEVEL}.npy"

    flat_full = resize_plane(flatfield, shape)
    flat_path = output_folder / f"flatfield_{label}{suffix}"
    np.save(flat_path, flat_full)
    imsave(
        flat_path.with_suffix(".png"),
        flat_full,
        cmap="coolwarm",
        vmin=0.4,
        vmax=1.6,
    )

    dark_full = match_darkfield(dark, shape)
    dark_path = output_folder / f"darkfield_{label}{suffix}"
    np.save(dark_path, dark_full)
    imsave(dark_path.with_suffix(".png"), dark_full, cmap="gray")

    factor = shape[0] / flatfield.shape[0] if flatfield.shape[0] else None
    logger.info(
        "  Upsampled to level %s %s (from level %s, x%.3g): %s, %s",
        FULL_RESOLUTION_LEVEL,
        tuple(shape),
        args.pyramid_level,
        factor,
        flat_path.name,
        dark_path.name,
    )
    return {
        "level": FULL_RESOLUTION_LEVEL,
        "shape": list(shape),
        "scale_factor": factor,
        "flatfield_path": str(flat_path),
        "darkfield_path": str(dark_path),
        "probed_tile": tile_name,
    }


def _write_products(
    label: str,
    output_folder: Path,
    result: FitResult,
    dark_plane: np.ndarray,
    stack: FitStack,
    tile_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """
    Save the fitted arrays and, when asked for, the figures.

    The saved darkfield is the pedestal actually subtracted before
    fitting, not BaSiC's own darkfield estimate: ``get_darkfield`` is
    off, so that estimate is all zeros and the pedestal is what an apply
    step needs.

    Parameters
    ----------
    label : str
        Name used for the output files and figures.
    output_folder : Path
        Folder to write into.
    result : FitResult
        The accepted fit.
    dark_plane : np.ndarray
        Darkfield plane subtracted before fitting.
    stack : FitStack
        Planes the fit was built from, needed by the figures.
    tile_names : list of str
        Tiles that contributed to the fit, in stack order.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict or None
        The validation metrics, or None when ``--validate`` was not
        passed.
    """
    np.save(output_folder / f"flatfield_{label}.npy", result.flatfield)
    np.save(output_folder / f"darkfield_{label}.npy", dark_plane)
    if not args.validate:
        return None
    return save_validation_figures(
        label,
        result.flatfield,
        dark_plane,
        stack,
        tile_names,
        output_folder / label,
        args.pyramid_level,
        args.validate_tiles,
    )


def estimate_dataset(
    label: str,
    tile_names: list[str],
    dark: np.ndarray | float,
    args: argparse.Namespace,
    output_folder: Path,
    basic_config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimate and save one flatfield from a folder of tiles.

    Parameters
    ----------
    label : str
        Name used for the output files and figure titles.
    tile_names : list of str
        Every tile to fit, all from the same channel folder.
    dark : np.ndarray or float
        Darkfield image or scalar pedestal.
    args : argparse.Namespace
        Parsed command-line arguments.
    output_folder : Path
        Folder the flatfield, darkfield and sidecar are written to.
    basic_config : dict
        BaSiC solver configuration for this run.
    provenance : dict
        Code, environment and argument provenance for this run.

    Returns
    -------
    dict
        The sidecar metadata written alongside the flatfield.
    """
    logger.info("%s", "=" * 60)
    logger.info("  %s (%d tiles)", label, len(tile_names))
    logger.info("%s", "=" * 60)

    # Loads the data from the tiles
    stack = load_fit_stack(
        args.base_path, tile_names, args.pyramid_level, args.max_fit_planes
    )
    # Matches the darkfield with the dimension of the stack
    dark_plane = match_darkfield(dark, (stack.height, stack.width))
    # Subtract the darkfield from the stack slices to estimate the flat
    fit_slices, pedestal = subtract_pedestal(stack.slices, dark_plane)
    scores = plane_scores(fit_slices)
    checks = prefit_checks(fit_slices, dark_plane, scores)
    plane_filter: dict[str, Any] = {
        "enabled": bool(args.drop_concentrated_planes),
        "applied": False,
    }
    if args.drop_concentrated_planes and checks["concentrated_signal"]:
        fit_slices, stack, report = drop_concentrated_planes(
            fit_slices, stack, scores
        )
        plane_filter.update(report, applied=True)
    elif args.drop_concentrated_planes:
        plane_filter["reason"] = (
            "the pre-fit check found no concentrated signal"
        )
        logger.info(
            "  Plane filter: on, but the pre-fit check found no concentrated "
            "signal - every plane is kept"
        )

    params, search_report, confirm = select_parameters(
        fit_slices, stack.z_offsets, args, basic_config
    )
    result = fit_with_guard(
        fit_slices,
        stack,
        args.method,
        params,
        (search_report or {}).get("baseline_std"),
        basic_config,
    )
    result, scaling = apply_flatfield_scaling(result, args)

    validation = _write_products(
        label, output_folder, result, dark_plane, stack, tile_names, args
    )

    upsampled = None
    if args.upsample:
        upsampled = upsample_to_full_resolution(
            label,
            output_folder,
            result.flatfield,
            dark,
            tile_names[0],
            args,
        )

    manifest_path = write_tile_manifest(label, output_folder, stack)
    sidecar = {
        "label": label,
        "base_path": args.base_path,
        "tile_pattern": args.tile_pattern,
        "params": result.params,
        "basic_config": basic_config,
        "method": args.method,
        "checks": checks,
        "plane_filter": plane_filter,
        "search": search_report,
        "confirm": confirm,
        # The guard's view: the mean-1 field before any rescaling.
        "flatfield_stats": result.stats,
        "guard": {"ok": bool(result.ok), "reasons": result.reasons},
        "scaling": scaling,
        "suspicious": (not result.ok),
        "n_tiles": len(tile_names),
        "planes_per_tile": int(stack.n_planes),
        "n_fit_images": int(fit_slices.shape[0]),
        "stack": summarize_stack(stack, fit_slices.shape[0], pedestal),
        "pyramid_level": args.pyramid_level,
        "darkfield_image": args.darkfield_image,
        "darkfield_value": (
            None if args.darkfield_image else float(args.darkfield_value)
        ),
        "darkfield_mean": float(dark_plane.mean()),
        "pedestal": pedestal,
        "n_z_fits_ok": result.n_z_fits_ok,
        "validation": validation,
        "upsampled": upsampled,
        "provenance": provenance,
        "metadata_files": {
            "run_log": f"{METADATA_DIR}/{RUN_LOG_NAME}",
            "tile_manifest": manifest_path,
        },
        # The only record of which tiles produced this flatfield: the
        # names are not recoverable from anything else.
        "tiles": tile_names,
    }
    sidecar_path = output_folder / f"flatfield_{label}.json"
    with open(sidecar_path, "w") as handle:
        json.dump(sidecar, handle, indent=2, default=str)
    logger.info(
        "  Saved flatfield_%s.npy, darkfield_%s.npy and " "flatfield_%s.json",
        label,
        label,
        label,
    )
    return sidecar


if __name__ == "__main__":
    # Kept so that ``python -m ...basicpy.estimate`` still runs the CLI.
    from aind_flatfield_correction.core.basicpy.cli import main

    main()
