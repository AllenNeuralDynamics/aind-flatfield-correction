"""
Per-pixel darkfield from the darkest planes of the darkest tiles.

On dim channels the darkfield decides the flatfield: the flatfield error
is about the darkfield error divided by the signal (the estimator's
pre-fit check reports it).  A scalar darkfield leaves the camera pattern
-- the sensor halves and the row stripes -- in the flatfield.  Dark frames
recorded with the light off are the best darkfield.  Without them, the
planes with the least light are the best estimate of the camera alone.

This module finds the darkest plane of every tile at a coarse pyramid
level, keeps the tiles whose darkest plane is darkest, reads the
full-resolution planes under those planes and takes the per-pixel median.

Planes in front of the tissue still receive scattered light, and its
amount changes from tile to tile.  A darkfield made from planes with light
is too high for the tiles with less light and removes real signal there.
The darkest tiles keep that light to a minimum, and two checks show what
is left:

* the spread of the darkest-plane level across tiles: a camera dark is the
  same in every tile, so a large spread means the dark planes contain
  light;
* the share of tiles whose darkest plane is below the darkfield's mean
  (by more than a small tolerance): for those tiles the darkfield is
  certainly too high.

Usage::

    python -m aind_flatfield_correction.core.basicpy.darkmap \\
        s3://bucket/dataset/SPIM --tile-pattern '_ch_488\\.ome\\.zarr$' \\
        --output-name ch_488 --output-folder ./darkmaps

It writes ``darkfield_<name>.npy`` at full resolution, a PNG preview and
the sidecar ``darkfield_<name>.json``.  Pass the ``.npy`` to the
estimator with ``--darkfield-image``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.image import imsave

from aind_flatfield_correction.core.basicpy.config import (
    DARKMAP_N_TILES,
    LEVEL_SUBSAMPLE,
    LIGHT_SPREAD_WARN,
    OVER_SUBTRACTED_TOLERANCE,
    OVER_SUBTRACTED_WARN,
)
from aind_flatfield_correction.core.basicpy.estimate import default_label
from aind_flatfield_correction.core.basicpy.tiles import (
    FULL_RESOLUTION_LEVEL,
    list_tiles,
    open_tile,
)
from aind_flatfield_correction.provenance import (
    attach_run_log,
    detach_run_log,
    gather_provenance,
    utc_now,
)

logger = logging.getLogger(__name__)

METADATA_DIR = "metadata"

RUN_LOG_NAME = "darkmap.log"


def plane_levels(
    base_path: str,
    tiles: list[str],
    level: int | str,
    subsample: int = LEVEL_SUBSAMPLE,
) -> list[np.ndarray]:
    """
    Measure the mean of every plane of every tile at a coarse level.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tiles.
    tiles : list of str
        Tile names.
    level : int or str
        Coarse pyramid level to read.
    subsample : int, optional
        Pixel step used for the means, by default
        :data:`~aind_flatfield_correction.core.basicpy.config.LEVEL_SUBSAMPLE`.

    Returns
    -------
    list of np.ndarray
        One array of plane means per tile, in Z order.
    """
    levels = []
    for index, tile in enumerate(tiles):
        arr = open_tile(base_path, tile, level)
        planes = np.asarray(
            arr[:, ::subsample, ::subsample].compute(), dtype=np.float64
        )
        levels.append(planes.reshape(planes.shape[0], -1).mean(axis=1))
        if (index + 1) % 10 == 0 or index + 1 == len(tiles):
            logger.info("    %d/%d tiles measured", index + 1, len(tiles))
    return levels


def darkest_tiles(
    levels: list[np.ndarray], n_tiles: int
) -> list[tuple[int, int, float]]:
    """
    Pick the tiles whose darkest plane is darkest.

    Parameters
    ----------
    levels : list of np.ndarray
        Plane means of every tile (:func:`plane_levels`).
    n_tiles : int
        Number of tiles to keep; all of them when there are fewer.

    Returns
    -------
    list of tuple of (int, int, float)
        Tile index, index of its darkest plane and that plane's mean,
        darkest first.
    """
    darkest = [
        (index, int(np.argmin(means)), float(np.min(means)))
        for index, means in enumerate(levels)
    ]
    return sorted(darkest, key=lambda item: item[2])[:n_tiles]


def light_spread(darkest_levels: np.ndarray) -> dict[str, float]:
    """
    Summarize how much the darkest-plane level changes across tiles.

    Parameters
    ----------
    darkest_levels : np.ndarray
        Darkest plane mean of every tile.

    Returns
    -------
    dict
        The 5th percentile, median and 95th percentile of the levels, and
        ``spread``, the 95th minus the 5th percentile.
    """
    p5, p50, p95 = np.percentile(darkest_levels, [5, 50, 95])
    return {
        "p5": float(p5),
        "median": float(p50),
        "p95": float(p95),
        "spread": float(p95 - p5),
    }


def full_resolution_median(
    base_path: str, tile: str, plane: int, coarse_depth: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Take the per-pixel median of the full-resolution planes under a plane.

    The planes under coarse plane ``z`` are ``z * f`` to ``(z + 1) * f``,
    with ``f`` the ratio of the full-resolution depth to the coarse depth
    (1 when the pyramid keeps the depth).

    Parameters
    ----------
    base_path : str
        Dataset path holding the tile.
    tile : str
        Tile name.
    plane : int
        Index of the plane at the coarse level.
    coarse_depth : int
        Number of planes of the tile at the coarse level.

    Returns
    -------
    tuple of (np.ndarray, dict)
        The median plane, and the planes read with their mean and their
        mean temporal variance (light adds photon noise to it; the camera
        offset adds none).
    """
    arr = open_tile(base_path, tile, FULL_RESOLUTION_LEVEL)
    factor = max(1, int(round(arr.shape[0] / coarse_depth)))
    start = plane * factor
    end = min(start + factor, arr.shape[0])
    planes = np.asarray(arr[start:end].compute(), dtype=np.float32)
    info = {
        "tile": tile,
        "full_resolution_planes": [int(start), int(end)],
        "mean": float(planes.mean()),
        "temporal_variance": (
            float(planes.var(axis=0, ddof=1).mean())
            if planes.shape[0] > 1
            else None
        ),
    }
    return np.median(planes, axis=0), info


def build_darkmap(
    base_path: str,
    tiles: list[str],
    level: int | str,
    n_tiles: int = DARKMAP_N_TILES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build the darkfield from the darkest planes of the darkest tiles.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tiles.
    tiles : list of str
        Tile names of one channel.
    level : int or str
        Coarse pyramid level used to find the darkest planes.
    n_tiles : int, optional
        Number of darkest tiles to use, by default
        :data:`~aind_flatfield_correction.core.basicpy.config.DARKMAP_N_TILES`.

    Returns
    -------
    tuple of (np.ndarray, dict)
        The full-resolution darkfield, and a report of the tiles and planes
        used, the checks and their warnings.
    """
    logger.info("  Measuring every plane of %d tiles", len(tiles))
    levels = plane_levels(base_path, tiles, level)
    darkest_all = np.array([float(np.min(means)) for means in levels])
    chosen = darkest_tiles(levels, n_tiles)

    medians, groups = [], []
    for index, plane, value in chosen:
        median, info = full_resolution_median(
            base_path, tiles[index], plane, len(levels[index])
        )
        info.update({"coarse_plane": plane, "coarse_plane_mean": value})
        medians.append(median)
        groups.append(info)
    darkmap = np.median(np.stack(medians), axis=0).astype(np.float32)

    map_mean = float(darkmap.mean())
    spread = light_spread(darkest_all)
    excess = map_mean - darkest_all
    over_subtracted = float(np.mean(excess > OVER_SUBTRACTED_TOLERANCE))
    report: dict[str, Any] = {
        "n_tiles_measured": len(tiles),
        "n_tiles_used": len(chosen),
        "coarse_level": level,
        "tiles_used": groups,
        "darkest_plane_per_tile": {
            tile: float(value) for tile, value in zip(tiles, darkest_all)
        },
        "darkest_plane_spread": spread,
        "over_subtracted_tiles": over_subtracted,
        "darkfield_stats": {
            "mean": map_mean,
            "median": float(np.median(darkmap)),
            "std": float(darkmap.std()),
            "min": float(darkmap.min()),
            "max": float(darkmap.max()),
        },
        "warnings": [],
    }
    if spread["spread"] > LIGHT_SPREAD_WARN:
        report["warnings"].append(
            f"the dark planes contain light: their level changes by "
            f"{spread['spread']:.1f} counts across tiles (a camera dark is "
            "the same in every tile); this darkfield keeps the light low by "
            "using the darkest tiles, recorded dark frames would remove it"
        )
    if over_subtracted > OVER_SUBTRACTED_WARN:
        report["warnings"].append(
            f"the darkfield is above the darkest plane of "
            f"{over_subtracted:.0%} of the tiles, so it removes real signal "
            "there"
        )
    for message in report["warnings"]:
        logger.warning("  Darkfield check: %s", message)
    logger.info(
        "  Darkfield from %d tiles: mean %.2f, pattern std %.2f counts; "
        "darkest-plane spread %.2f counts; %.0f%% of tiles below it",
        len(chosen),
        map_mean,
        report["darkfield_stats"]["std"],
        spread["spread"],
        100 * over_subtracted,
    )
    return darkmap, report


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the darkfield entry point.
    """
    parser = argparse.ArgumentParser(
        description="Darkfield from the darkest planes of the darkest tiles."
    )
    parser.add_argument(
        "base_path",
        type=str,
        help="Folder holding one channel's tiles. Accepts s3 input.",
    )
    parser.add_argument(
        "--tile-pattern",
        type=str,
        default=None,
        help="Regular expression used to keep only the tiles whose name "
        r"matches it, e.g. '_ch_488\.ome\.zarr$'.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name used for the output files. Defaults to the last "
        "segment of base_path.",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=3,
        help="Pyramid level used to find the darkest planes (default 3). "
        "The darkfield itself is built at full resolution.",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=DARKMAP_N_TILES,
        help="Number of darkest tiles to use " f"(default {DARKMAP_N_TILES}).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="darkfield_estimation",
        help="Folder to save the darkfield in.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Build a darkfield and write it with its sidecar.

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
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    started = utc_now()
    output_folder = Path(args.output_folder)
    (output_folder / METADATA_DIR).mkdir(parents=True, exist_ok=True)
    log_handler = attach_run_log(output_folder / METADATA_DIR / RUN_LOG_NAME)
    try:
        pattern = re.compile(args.tile_pattern) if args.tile_pattern else None
        tiles = list_tiles(args.base_path, pattern)
        label = args.output_name or default_label(args.base_path)
        logger.info("Found %d tiles | output name: %s", len(tiles), label)

        darkmap, report = build_darkmap(
            args.base_path, tiles, args.pyramid_level, args.n_tiles
        )
        map_path = output_folder / f"darkfield_{label}.npy"
        np.save(map_path, darkmap)
        low, high = np.percentile(darkmap, [1, 99])
        imsave(
            map_path.with_suffix(".png"),
            np.clip(darkmap, low, high),
            cmap="gray",
        )
        sidecar = {
            "label": label,
            "base_path": args.base_path,
            "tile_pattern": args.tile_pattern,
            "darkfield_path": str(map_path),
            **report,
            "provenance": gather_provenance(args, started, utc_now()),
            "metadata_files": {"run_log": f"{METADATA_DIR}/{RUN_LOG_NAME}"},
        }
        with open(output_folder / f"darkfield_{label}.json", "w") as handle:
            json.dump(sidecar, handle, indent=2, default=str)
        logger.info("  Saved darkfield_%s.npy, .png and .json", label)
    finally:
        detach_run_log(log_handler)


if __name__ == "__main__":
    main()
