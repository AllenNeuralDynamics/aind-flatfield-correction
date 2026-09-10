"""
Estimate flatfields using BaSiC.

One run fits one multiplicative flatfield from a bounded number of Z
planes taken from every tile under ``base_path``.  The sensor camera
offset is removed before fitting.

``base_path`` is one channel's folder: nothing is inferred from the tile
names, so to estimate several channels the command is run once per
channel folder.  ``--tile-pattern`` optionally narrows the run to a
subset of the tiles inside that folder.

Examples
--------
Estimate one channel of a public SmartSPIM dataset::

    python -m aind_flatfield_correction.core.basicpy.estimate \\
        s3://aind-open-data/HCR_800792_2026-03-25_13-00-00/SPIM/ch_405 \\
        --pyramid-level 3 \\
        --darkfield-value 90 \\
        --output-folder /results/flatfields

Fit a subset of a flat tile folder, fitting every Z plane independently
and combining by pixelwise median, and write the inspection figures::

    python -m aind_flatfield_correction.core.basicpy.estimate <base_path> \\
        --tile-pattern '_ch_405\\.ome\\.zarr$' --output-name ch405 \\
        --method per-z-median --validate --output-folder ./flatfields

Override part of the solver configuration, inline or from a file.  The
given keys are merged over the built-in defaults::

    python -m aind_flatfield_correction.core.basicpy.estimate <base_path> \\
        --basic-config '{"fitting_mode": "approximate"}'

    python -m aind_flatfield_correction.core.basicpy.estimate <base_path> \\
        --basic-config ./basic_config.json

``smoothness_flatfield`` is supplied the same way.  It is the parameter
the search varies, so the configured value is the incumbent every
candidate is measured against; with ``--skip-search`` it is fitted
directly::

    python -m aind_flatfield_correction.core.basicpy.estimate <base_path> \\
        --basic-config '{"smoothness_flatfield": 2.5}' --skip-search
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from aind_flatfield_correction.core.basicpy.config import (
    baseline_params,
    load_basic_config,
)
from aind_flatfield_correction.core.basicpy.figures import (
    save_validation_figures,
)
from aind_flatfield_correction.core.basicpy.fit import (
    estimate_joint,
    estimate_per_z_median,
    report_flatfield,
    validate_basic_config,
)
from aind_flatfield_correction.core.basicpy.search import (
    confirm_against_baseline,
    parallel_autotune,
)
from aind_flatfield_correction.core.basicpy.tiles import (
    FULL_RESOLUTION_LEVEL,
    FitStack,
    list_tiles,
    load_darkfield,
    load_fit_stack,
    match_darkfield,
    probe_plane_shape,
    resize_plane,
    subtract_pedestal,
)
from aind_flatfield_correction.provenance import (
    attach_run_log,
    detach_run_log,
    gather_provenance,
    utc_now,
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
        The fitted flatfield, at the estimation level.
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

    dark_full = match_darkfield(dark, shape)
    dark_path = output_folder / f"darkfield_{label}{suffix}"
    np.save(dark_path, dark_full)

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
        "search": search_report,
        "confirm": confirm,
        "flatfield_stats": result.stats,
        "guard": {"ok": bool(result.ok), "reasons": result.reasons},
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


def default_label(base_path: str) -> str:
    """
    Derive an output name from the channel folder's own name.

    Parameters
    ----------
    base_path : str
        Channel folder, ``s3://`` or local.

    Returns
    -------
    str
        The last non-empty path segment, with anything but letters,
        digits, dot, dash and underscore replaced by an underscore.
        Falls back to ``"flatfield"`` for a path with no usable
        segment.
    """
    segments = [
        segment
        for segment in base_path.replace("s3://", "").split("/")
        if segment
    ]
    if not segments:
        return "flatfield"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", segments[-1]).strip("_")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the input, darkfield and output arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    None
    """
    parser.add_argument(
        "base_path",
        type=str,
        help="Folder holding one channel's tiles. Accepts s3 input. "
        "Every OME-Zarr inside it is fitted together, so run the "
        "command once per channel folder.",
    )
    parser.add_argument(
        "--tile-pattern",
        type=str,
        default=None,
        help="Regular expression used to keep only the tiles whose name "
        r"matches it, e.g. '_ch_405\.ome\.zarr$'. Default: every entry "
        "in the folder, which must then hold only tiles.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name used for the output files and figures. Defaults to "
        "the last segment of base_path, e.g. 'ch_405'.",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=3,
        help="Pyramid level to use for the estimation (default 3).",
    )
    parser.add_argument(
        "--darkfield-image",
        type=str,
        default=None,
        help="Darkfield image (.npy) to use for the estimation.",
    )
    parser.add_argument(
        "--darkfield-value",
        type=float,
        default=0.0,
        help="Darkfield value to use for the estimation. "
        "This is ignored if darkfield-image is provided.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="flatfield_estimation",
        help="Folder to save the output flatfield estimations.",
    )


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the estimation-method and validation arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--method",
        choices=["fit", "per-z-median"],
        default="fit",
        help="fit (default): one joint BaSiC fit over all tiles x a "
        "bounded number of Z planes each.  per-z-median: one fit per Z "
        "index across all tiles, combined with a pixelwise median.",
    )
    parser.add_argument(
        "--basic-config",
        type=str,
        default=None,
        help="BaSiC solver configuration, as a path to a JSON file or "
        "an inline JSON object. Merged over the built-in known-good "
        "defaults, so only the keys being changed need to be given. "
        "smoothness_flatfield is the parameter the search varies: the "
        "value given here is the incumbent it must beat, and the value "
        "fitted directly under --skip-search.",
    )
    parser.add_argument(
        "--max-fit-planes",
        type=int,
        default=2000,
        help="Image budget for the joint fit (default 2000). "
        "Planes per tile = budget // n_tiles, taken from the middle "
        "60%% of Z.",
    )
    parser.add_argument(
        "--upsample",
        dest="upsample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the flatfield and darkfield resampled from the "
        "estimation pyramid level up to the full-resolution level, "
        "whose extent is read from the dataset metadata (default: on; "
        "disable with --no-upsample).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Emit X/Y intensity-profile plots for a few representative "
        "tiles so each flatfield can be inspected and accepted before "
        "use.",
    )
    parser.add_argument(
        "--validate-tiles",
        type=int,
        default=4,
        help="Number of tiles to plot profiles for when --validate is "
        "set (default 4, spread across the mosaic).",
    )


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the parameter-search arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip the parameter search and use the known-good manual "
        "params.",
    )
    parser.add_argument(
        "--max-eval-slices",
        type=int,
        default=150,
        help="Slices used to score each search candidate (default 150).",
    )
    parser.add_argument(
        "--n-confirm",
        type=int,
        default=1500,
        help="Slices for the confirmation refit (default 1500; 0 "
        "disables).",
    )
    parser.add_argument(
        "--max-search-minutes",
        type=float,
        default=120.0,
        help="Abort the search and use manual params if projected to "
        "exceed this.",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the estimation entry point.
    """
    parser = argparse.ArgumentParser(description="BaSiC flatfield estimation.")
    _add_data_arguments(parser)
    _add_fit_arguments(parser)
    _add_search_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Main function for BaSiC flatfield estimation.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments to parse. If None, uses sys.argv.

    Returns
    -------
    None
    """
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Keep JAX on CPU here and in every spawned worker.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # The run log is attached before the first record, so that the file
    # is a complete account of the run rather than missing its opening
    # line.
    started = utc_now()
    output_folder = Path(args.output_folder)
    (output_folder / METADATA_DIR).mkdir(parents=True, exist_ok=True)
    log_handler = attach_run_log(output_folder / METADATA_DIR / RUN_LOG_NAME)
    logger.info("Starting BaSiC flatfield estimation with arguments: %s", args)

    # Resolved and checked before anything expensive: an unusable
    # solver configuration should not surface after a long tile load.
    basic_config = load_basic_config(args.basic_config)
    validate_basic_config(basic_config)

    pattern = re.compile(args.tile_pattern) if args.tile_pattern else None
    tile_names = list_tiles(args.base_path, pattern)
    label = args.output_name or default_label(args.base_path)
    logger.info("Found %d tiles | output name: %s", len(tile_names), label)

    dark = load_darkfield(args.darkfield_image, args.darkfield_value)
    try:
        estimate_dataset(
            label,
            tile_names,
            dark,
            args,
            output_folder,
            basic_config,
            gather_provenance(args, started, utc_now()),
        )
        logger.info(
            "Ending BaSiC flatfield estimation with outputs at: %s",
            output_folder,
        )
    finally:
        # Detached in a finally so a run that raises still leaves a
        # complete, flushed log behind.
        detach_run_log(log_handler)


if __name__ == "__main__":
    main()
