"""
Command-line entry point for BaSiC flatfield estimation.

One run fits one multiplicative flatfield from a bounded number of Z
planes taken from every tile under ``base_path``.  The sensor camera
offset is removed before fitting.

The saved flatfield is rescaled so that its 99.9th percentile
(``--scale-percentile``), rather than its mean, is 1.0: dividing by
basicpy's mean-1 field would dim the bright centre of every tile.
``--no-scale-flatfield`` keeps the mean-1 field.

``base_path`` is one channel's folder: nothing is inferred from the tile
names, so to estimate several channels the command is run once per
channel folder.  ``--tile-pattern`` optionally narrows the run to a
subset of the tiles inside that folder.

Examples
--------
Estimate one channel of a public SmartSPIM dataset::

    python -m aind_flatfield_correction.core.basicpy.cli \\
        s3://aind-open-data/HCR_800792_2026-03-25_13-00-00/SPIM/ch_405 \\
        --pyramid-level 3 \\
        --darkfield-value 90 \\
        --output-folder /results/flatfields

Fit a subset of a flat tile folder, fitting every Z plane independently
and combining by pixelwise median, and write the inspection figures::

    python -m aind_flatfield_correction.core.basicpy.cli <base_path> \\
        --tile-pattern '_ch_405\\.ome\\.zarr$' --output-name ch405 \\
        --method per-z-median --validate --output-folder ./flatfields

Override part of the solver configuration, inline or from a file.  The
given keys are merged over the built-in defaults::

    python -m aind_flatfield_correction.core.basicpy.cli <base_path> \\
        --basic-config '{"fitting_mode": "approximate"}'

    python -m aind_flatfield_correction.core.basicpy.cli <base_path> \\
        --basic-config ./basic_config.json

``smoothness_flatfield`` is supplied the same way.  It is the parameter
the search varies, so the configured value is the incumbent every
candidate is measured against; with ``--skip-search`` it is fitted
directly::

    python -m aind_flatfield_correction.core.basicpy.cli <base_path> \\
        --basic-config '{"smoothness_flatfield": 2.5}' --skip-search
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from aind_flatfield_correction.core.basicpy.config import (
    FLAT_SCALE_PERCENTILE,
    load_basic_config,
)
from aind_flatfield_correction.core.basicpy.estimate import (
    METADATA_DIR,
    RUN_LOG_NAME,
    estimate_dataset,
)
from aind_flatfield_correction.core.basicpy.fit import validate_basic_config
from aind_flatfield_correction.core.basicpy.tiles import (
    default_label,
    list_tiles,
    load_darkfield,
)
from aind_flatfield_correction.provenance import (
    attach_run_log,
    detach_run_log,
    gather_provenance,
    utc_now,
)

logger = logging.getLogger(__name__)


def _percentile(text: str) -> float:
    """
    Parse a percentile argument, rejecting values outside (0, 100].

    Checked at parse time so that a typo fails before the tile load
    rather than after the fit.

    Parameters
    ----------
    text : str
        The value as given on the command line.

    Returns
    -------
    float
        The percentile.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a number in (0, 100].
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < value <= 100.0:
        raise argparse.ArgumentTypeError(
            f"percentile must be in (0, 100], got {value:g}"
        )
    return value


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
        "--drop-concentrated-planes",
        action="store_true",
        help="When the pre-fit check finds concentrated signal, drop the "
        "planes that are bright (mean above 3x the median plane) and have "
        "one dominant bright region (99th percentile of the background map "
        "above 3x its median) before the fit, so the flatfield does not "
        "learn a bright tissue region.  Healthy channels keep every plane.  "
        "Off by default; only with --method fit.",
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
        "--scale-flatfield",
        dest="scale_flatfield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale the saved flatfield so that its --scale-percentile, "
        "rather than its mean, is 1.0, so that dividing by it no longer "
        "dims the bright centre of the tile (default: on; keep basicpy's "
        "mean-1 flatfield with --no-scale-flatfield).",
    )
    parser.add_argument(
        "--scale-percentile",
        type=_percentile,
        default=FLAT_SCALE_PERCENTILE,
        help="Percentile of the flatfield that --scale-flatfield maps to "
        f"1.0, in (0, 100] (default {FLAT_SCALE_PERCENTILE:g}).",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    # The per-Z method fits each Z index across all tiles, so dropping
    # single planes would leave those groups uneven.
    if args.drop_concentrated_planes and args.method != "fit":
        parser.error("--drop-concentrated-planes works only with --method fit")

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
