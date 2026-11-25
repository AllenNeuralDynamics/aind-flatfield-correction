"""
Tile discovery and plane loading for flatfield estimation.
It also includes utilities for handling darkfield images and
resizing them for flatfield estimation as well as subtracting
darkfield images from the raw data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple, Optional

import boto3
import numpy as np
from aind_large_scale_prediction.io import OMEZarrReader
from botocore import UNSIGNED
from botocore.client import Config
from skimage.filters import gaussian
from skimage.transform import resize as sk_resize
from tifffile import imread as tif_imread

from aind_flatfield_correction.core.basicpy.config import (
    BG_GRID_BLOCKS,
    BRIGHT_RATIO,
    LOCALIZED_MAX,
)

logger = logging.getLogger(__name__)

# OME-NGFF orders its multiscale datasets from highest to lowest
# resolution, so the first one is the full-resolution level.
FULL_RESOLUTION_LEVEL = "0"
ZARR_VERSIONS = ("2.0", "3.0")


class TileRecord(NamedTuple):
    """
    What one tile contributed to the fit.

    Kept so that a finished run can say exactly which planes it was
    built from, and so that a dead tile (``mean`` near zero) or a tile
    smaller than the rest (``padded``) is visible after the fact.
    """

    name: str
    shape: tuple[int, ...]
    z_indices: tuple[int, ...]
    mean: float
    padded: bool


class FitStack(NamedTuple):
    """
    Raw planes gathered from every tile of one channel.

    ``z_offsets`` maps a tile's index in the tile-name list to its
    ``(start, end)`` span in ``slices``; ``manifest`` holds one
    :class:`TileRecord` per tile, in the same order.
    """

    slices: np.ndarray
    z_offsets: dict[int, tuple[int, int]]
    n_planes: int
    height: int
    width: int
    manifest: tuple[TileRecord, ...] = ()


def _list_tile_names_s3(base_path: str) -> list[str]:
    """
    List the immediate child prefixes of an ``s3://`` dataset path.

    Parameters
    ----------
    base_path : str
        Dataset path of the form ``s3://bucket/prefix``.

    Returns
    -------
    list of str
        Names of the prefixes directly under ``base_path``.
    """
    without_scheme = base_path.removeprefix("s3://")
    bucket, _, prefix = without_scheme.partition("/")
    # Unsigned: AIND acquisition data is served from public buckets.
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix.rstrip("/") + "/",
        Delimiter="/",
    )
    names = []
    for page in pages:
        for common in page.get("CommonPrefixes", []):
            names.append(common["Prefix"].rstrip("/").split("/")[-1])
    return names


def _list_tile_names_local(base_path: str) -> list[str]:
    """
    List the entries of a local dataset directory.

    Parameters
    ----------
    base_path : str
        Path to a directory holding the tile OME-Zarrs.

    Returns
    -------
    list of str
        Sorted entry names.

    Raises
    ------
    FileNotFoundError
        If ``base_path`` is not a directory.
    """
    root = Path(base_path)
    if not root.is_dir():
        raise FileNotFoundError(f"base_path is not a directory: {base_path}")
    return sorted(entry.name for entry in root.iterdir())


def list_tiles(
    base_path: str,
    tile_pattern: re.Pattern | None = None,
) -> list[str]:
    r"""
    Discover every tile under a dataset path.

    ``base_path`` is one channel's folder: every OME-Zarr directly
    inside it is taken to be a tile of the same channel.  Nothing is
    inferred from the tile names -- pass ``tile_pattern`` to select a
    subset, e.g. ``re.compile(r"_ch_405\.ome\.zarr$")``.

    Parameters
    ----------
    base_path : str
        Channel folder, either ``s3://bucket/prefix`` or a local
        directory.
    tile_pattern : re.Pattern, optional
        Regular expression searched against each name. Without it every
        entry is taken, so the folder must hold only tiles.

    Returns
    -------
    list of str
        Sorted tile names.

    Raises
    ------
    ValueError
        If no entry is left to fit.
    """
    names = (
        _list_tile_names_s3(base_path)
        if base_path.startswith("s3://")
        else _list_tile_names_local(base_path)
    )
    if tile_pattern is None:
        # The user guarantees the folder holds only tile OME-Zarrs.
        tiles = list(names)
    else:
        # search(), not match(): the pattern is a filter, so it should
        # also accept a fragment such as "_ch_405".
        tiles = [name for name in names if tile_pattern.search(name)]

    if not tiles:
        raise ValueError(
            f"No tiles left to fit under {base_path}"
            + (f" matching {tile_pattern.pattern!r}" if tile_pattern else "")
        )
    tiles.sort()
    return tiles


def open_tile(
    base_path: str,
    name: str,
    level: int | str,
    zarr_version: Optional[str] = None,
):
    """
    Open one tile as a 3-D ``(Z, Y, X)`` dask array.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tile.
    name : str
        Tile directory name.
    level : int or str
        Multiscale (pyramid) level to read.
    zarr_version : str, optional
        Zarr format version to use. If ``None``, the function will try
        both "3.0" and "2.0".

    Returns
    -------
    dask.array.Array
        Lazy ``(Z, Y, X)`` view of the tile; leading singleton axes are
        dropped.
    """
    uri = f"{base_path.rstrip('/')}/{name}"
    if zarr_version is None:
        failures = []
        for candidate in ZARR_VERSIONS:
            try:
                reader = OMEZarrReader(
                    data_path=uri,
                    multiscale=str(level),
                    zarr_version=candidate,
                )
                break
            except Exception as error:  # noqa: BLE001
                failures.append(f"zarr {candidate}: {error}")
                logger.debug(
                    "  tile %s did not open as zarr %s: %s",
                    name,
                    candidate,
                    error,
                )
        else:
            logger.error(
                "Failed to open tile %s at %s; tried %s",
                name,
                uri,
                " | ".join(failures),
            )
            raise ValueError(
                f"Failed to open tile {name} as any of "
                f"{', '.join(ZARR_VERSIONS)}"
            )
    else:
        try:
            reader = OMEZarrReader(
                data_path=uri,
                multiscale=str(level),
                zarr_version=zarr_version,
            )
        except Exception as error:  # noqa: BLE001
            logger.error("Failed to open tile %s at %s: %s", name, uri, error)
            raise ValueError(
                f"Failed to open tile {name} as zarr {zarr_version}"
            ) from error

    arr = reader.as_dask_array()
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def pick_z_planes(
    z_dim: int, n_planes: int, edge_fraction: float = 0.2
) -> np.ndarray:
    """
    Evenly spaced Z indices from the middle of the stack.

    The light sheet is well-behaved through the
    middle of the stack, while planes near either end are dominated by
    background and would bias the flatfield flatter.

    Parameters
    ----------
    z_dim : int
        Number of Z planes in the tile.
    n_planes : int
        Number of planes wanted.
    edge_fraction : float, optional
        Fraction of the stack excluded at each end, by default 0.2.

    Returns
    -------
    np.ndarray
        Unique, sorted Z indices.
    """
    start = int(z_dim * edge_fraction)
    end = z_dim - start
    if end - start < 1:  # degenerate / thin stack
        start, end = 0, z_dim
    n_planes = int(max(1, min(n_planes, end - start)))
    planes = np.linspace(start, end - 1, n_planes)
    return np.unique(planes.round().astype(int))


def probe_tile_shapes(
    base_path: str, tiles: list[str], level: int | str
) -> dict[str, tuple[int, ...]]:
    """
    Read every tile's shape at ``level`` without fetching voxels.

    Keyed by tile name rather than by shape: tiles of an acquisition
    usually share their ``(Y, X)`` extent, so keying by shape would
    collapse them into one entry and make the minimum Z wrong.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tiles.
    tiles : list of str
        Tile names to probe.
    level : int or str
        Multiscale level to probe.

    Returns
    -------
    dict
        ``{tile_name: (Z, H, W)}`` for each tile.
    """
    shapes = {}
    for i, tile in enumerate(tiles, 1):
        arr = open_tile(base_path, tile, level)
        shapes[tile] = tuple(arr.shape)  # assumes ZYX order
        if i % 25 == 0 or i == len(tiles):
            logger.info("    probed %d/%d tiles", i, len(tiles))
    return shapes


def load_fit_stack(
    base_path: str,
    tiles: list[str],
    level: int | str,
    max_planes: int,
) -> FitStack:
    """
    Stream a bounded number of Z planes from every tile into one stack.

    Every tile contributes.  What is bounded is the number of Z planes
    taken from each: N tiles at one Z index are N different regions seen
    through the same optics, and so are highly informative about the
    ``(y, x)`` profile, whereas consecutive Z planes of one tile are
    nearly redundant. The image budget is therefore spent on tiles, not
    on Z.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tiles.
    tiles : list of str
        Tile names of a single channel.
    level : int or str
        Multiscale level to read.
    max_planes : int
        Total image budget across all tiles.

    Returns
    -------
    FitStack
        The loaded planes, per-tile spans into them, the planes taken
        per tile, the padded plane height and width, and one
        :class:`TileRecord` per tile.
    """
    logger.info("  Probing tile shapes at level %s", level)
    shapes = probe_tile_shapes(base_path, tiles, level)
    max_h = max(shape[1] for shape in shapes.values())
    max_w = max(shape[2] for shape in shapes.values())
    min_z = min(shape[0] for shape in shapes.values())
    planes_per_tile = max(1, max_planes // len(tiles))
    n_planes = int(min(planes_per_tile, min_z))
    total = n_planes * len(tiles)
    logger.info(
        "  Tiles: %d  |  budget %d images  ->  %d planes/tile from the "
        "middle 60%% of Z  ->  %d images  |  plane %dx%d",
        len(tiles),
        max_planes,
        n_planes,
        total,
        max_h,
        max_w,
    )

    # Initializing the array
    stack = np.zeros((total, max_h, max_w), dtype=np.float32)

    # Spans are keyed by the tile's index in ``tiles``, which is all the
    # plane bookkeeping needs.  Laying the planes out on the acquisition
    # grid would need real stage positions: read them from the
    # acquisition.json metadata rather than inferring them from tile names.
    z_offsets: dict[int, tuple[int, int]] = {}
    manifest: list[TileRecord] = []

    offset = 0
    for index, tile in enumerate(tiles):
        arr = open_tile(base_path, tile, level)
        z_idx = pick_z_planes(arr.shape[0], n_planes)
        vol = np.asarray(arr[z_idx].compute(), dtype=np.float32)
        vol = vol[:, :max_h, :max_w]
        n_z, plane_h, plane_w = vol.shape
        end = offset + n_z
        stack[offset:end, :plane_h, :plane_w] = vol
        z_offsets[index] = (offset, end)
        manifest.append(
            TileRecord(
                name=tile,
                shape=shapes[tile],
                z_indices=tuple(int(z) for z in z_idx),
                mean=float(vol.mean()),
                padded=(plane_h, plane_w) != (max_h, max_w),
            )
        )
        offset = end
        del vol
        if (index + 1) % 10 == 0 or index + 1 == len(tiles):
            logger.info("    %d/%d tiles loaded", index + 1, len(tiles))

    stack = stack[:offset]
    logger.info(
        "  Stack: %s  min=%.1f  max=%.1f",
        stack.shape,
        float(stack.min()),
        float(stack.max()),
    )
    return FitStack(stack, z_offsets, n_planes, max_h, max_w, tuple(manifest))


def load_darkfield(
    darkfield_image: str | None, darkfield_value: float
) -> np.ndarray | float:
    """
    Resolve the darkfield camera offset from the command-line arguments.

    A darkfield saved as a stack of dark frames is reduced to one plane
    by averaging over the leading axes, which is how such a stack is
    meant to be combined: averaging N dark exposures is what suppresses
    read noise and leaves the fixed pedestal.  Only a 2-D plane can be
    subtracted from the fitting stack or resized to the estimation
    level.

    Parameters
    ----------
    darkfield_image : str or None
        Path to a ``.npy``, ``.tif`` or ``.tiff`` darkfield image.
        Takes precedence over ``darkfield_value``.
    darkfield_value : float
        Scalar pedestal in ADU counts, used when no image is given.

    Returns
    -------
    np.ndarray or float
        A 2-D darkfield plane, or the scalar pedestal.

    Raises
    ------
    Exception
        Whatever the reader raised, after logging the path; a darkfield
        that cannot be read must not silently become a no-op.
    """
    if darkfield_image:
        try:
            if darkfield_image.endswith(".tif") or darkfield_image.endswith(
                ".tiff"
            ):
                dark = tif_imread(darkfield_image).astype(np.float32)
            else:
                # Defaults to numpy array
                # Errors out if the file is not a valid numpy array
                dark = np.load(darkfield_image).astype(np.float32)
        except Exception as e:
            logger.error(
                "Failed to load darkfield image %s: %s", darkfield_image, e
            )
            raise

        if dark.ndim > 2:
            dark = dark.mean(axis=tuple(range(dark.ndim - 2)))
        logger.info(
            "  Darkfield image %s: shape=%s mean=%.2f",
            darkfield_image,
            dark.shape,
            float(dark.mean()),
        )
        return np.asarray(dark, dtype=np.float32)
    return float(darkfield_value)


def resize_plane(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Resample a 2-D plane onto ``shape``.

    Bilinear with edge padding and no anti-aliasing, matching how the
    correction is applied: these fields are smooth by construction, so
    interpolating them is exact enough, and anti-aliasing would only
    blur an already smooth surface.  ``preserve_range`` keeps the
    values in gain units instead of rescaling them to [0, 1].

    Parameters
    ----------
    plane : np.ndarray
        Source ``(H, W)`` plane.
    shape : tuple of int
        Target ``(height, width)``.

    Returns
    -------
    np.ndarray
        The resampled plane, float32. The input is returned unchanged
        when it is already the requested shape.
    """
    plane = np.asarray(plane, dtype=np.float32)
    if plane.shape == tuple(shape):
        return plane
    resized = sk_resize(
        plane,
        shape,
        order=1,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.asarray(resized, dtype=np.float32)


def probe_plane_shape(
    base_path: str,
    tile_name: str,
    level: int | str = FULL_RESOLUTION_LEVEL,
) -> tuple[int, int]:
    """
    Read one tile's plane extent at ``level`` from its metadata.

    Only the array's shape is touched, so this costs a metadata read
    and fetches no voxels.

    Parameters
    ----------
    base_path : str
        Dataset path holding the tile.
    tile_name : str
        Tile whose extent stands for the channel's.
    level : int or str, optional
        Multiscale level to read, by default the highest-resolution
        level.

    Returns
    -------
    tuple of int
        The ``(height, width)`` of a plane at that level.
    """
    arr = open_tile(base_path, tile_name, level)
    height, width = arr.shape[-2], arr.shape[-1]
    return int(height), int(width)


def match_darkfield(
    dark: np.ndarray | float, shape: tuple[int, int]
) -> np.ndarray:
    """
    Broadcast or resize the darkfield onto a ``(height, width)`` plane.

    A full-resolution darkfield that is an exact multiple of the target
    is averaged over blocks, because a pyramid level is the block mean of
    full resolution: the camera pattern changes from pixel to pixel, and
    sampling one pixel per block would subtract noise instead of the
    pattern the level holds.  Any other size is resized.

    Parameters
    ----------
    dark : np.ndarray or float
        Darkfield image or scalar pedestal.
    shape : tuple of int
        Target ``(height, width)`` at the estimation pyramid level.

    Returns
    -------
    np.ndarray
        Darkfield plane of shape ``shape``, float32.
    """
    if np.isscalar(dark):
        return np.full(shape, float(dark), dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    height, width = shape
    rows, cols = dark.shape[0] // height, dark.shape[1] // width
    if (
        dark.shape != tuple(shape)
        and rows >= 1
        and cols >= 1
        and dark.shape == (rows * height, cols * width)
    ):
        logger.info(
            "  Darkfield averaged over %dx%d blocks %s -> %s for the "
            "estimation level",
            rows,
            cols,
            dark.shape,
            tuple(shape),
        )
        blocks = dark.reshape(height, rows, width, cols)
        return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    # Resizing is necessary since we could have a high resolution
    # darkfield image but using a downsampled version for the estimation
    if dark.shape != tuple(shape):
        logger.info(
            "  Darkfield resized %s -> %s for the estimation level",
            dark.shape,
            tuple(shape),
        )
    return resize_plane(dark, shape)


def subtract_pedestal(
    slices: np.ndarray, dark_plane: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Remove the sensor pedestal before fitting.

    ``raw = pedestal + signal x flatfield``.  The offset is added by the
    sensor after light collection, so it is not vignetted, and a
    multiplicative field must be estimated on the signal alone.

    Parameters
    ----------
    slices : np.ndarray
        Raw ``(N, H, W)`` stack.
    dark_plane : np.ndarray
        Darkfield ``(H, W)`` plane to subtract.

    Returns
    -------
    tuple of (np.ndarray, dict)
        The pedestal-subtracted stack, and the raw and corrected
        medians plus the dilution factor that fitting on raw values
        would have introduced.
    """
    fit_slices = np.clip(slices - dark_plane, 0.0, None)
    med_raw = float(np.median(slices))
    med_fit = float(np.median(fit_slices))
    info = {
        "median_raw": med_raw,
        "median_pedestal_removed": med_fit,
        "dilution_if_fitted_on_raw": (
            med_fit / med_raw if med_raw else float("nan")
        ),
    }
    logger.info(
        "  Pedestal removed for fitting: median %.1f -> %.1f (fitting "
        "on raw would dilute the flatfield by x%.3f)",
        med_raw,
        med_fit,
        info["dilution_if_fitted_on_raw"],
    )
    return fit_slices, info


def plane_concentration(
    fit_slices: np.ndarray, hot_ratio: float = 5.0
) -> dict[str, float]:
    """
    Measure how unevenly the fit objective is spread across planes.

    BaSiC minimises a squared error weighted by absolute intensity, so a few
    very bright planes -- a tracer injection site, say -- can carry most of the
    objective and pull the flatfield towards their own content. This reports
    that imbalance so a caller can refuse the fit rather than trust it.

    Measured over three SmartSPIM acquisitions: healthy channels scored 0.144
    to 0.330, injection-contaminated ones 0.590 to 0.632. The figure is stable
    against the plane budget -- the same channel read 0.594 at 280 planes and
    0.577 at 1988.

    ``hot_fraction`` is reported separately and is *not* a substitute for
    ``top_share``: one healthy channel held 44 planes above the ratio while
    scoring only 0.199, because those planes were moderately bright rather
    than dominant.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.
    hot_ratio : float, optional
        Multiple of the median plane mean above which a plane counts as hot,
        by default 5.0.

    Returns
    -------
    dict
        ``top_share`` (fraction of the total sum-of-squares held by the
        brightest 3.6% of planes), ``top_n`` (how many planes that is),
        ``hot_fraction``, ``hot_planes`` and ``plane_mean_ratio`` (brightest
        plane mean over the median).
    """
    energy = (np.asarray(fit_slices, dtype=np.float64) ** 2).sum(axis=(1, 2))
    total = float(energy.sum())
    n_planes = int(energy.size)
    # 10 of 280 was the reference proportion these thresholds were measured at.
    top_n = max(1, round(n_planes * 10 / 280))
    ranked = np.sort(energy)[::-1]
    top_share = float(ranked[:top_n].sum() / total) if total else float("nan")

    plane_means = np.asarray(fit_slices, dtype=np.float64).mean(axis=(1, 2))
    median_mean = float(np.median(plane_means))
    hot = plane_means > hot_ratio * median_mean

    return {
        "top_share": top_share,
        "top_n": top_n,
        "n_planes": n_planes,
        "hot_planes": int(hot.sum()),
        "hot_fraction": float(hot.mean()),
        "plane_mean_ratio": (
            float(plane_means.max() / median_mean)
            if median_mean
            else float("nan")
        ),
    }


def background_map(
    plane: np.ndarray, n_blocks: int = BG_GRID_BLOCKS
) -> np.ndarray:
    """
    Map the diffuse background of one plane on a coarse grid.

    Each grid cell is the median of one block of pixels, so sparse bright
    cells do not change it and the map shows how the diffuse signal is
    spread over the field of view.  The block size comes from the short
    side of the plane, so that the same physical block is used at every
    pyramid level.  A Gaussian of one block then removes the remaining
    noise.

    Parameters
    ----------
    plane : np.ndarray
        One pedestal-subtracted ``(H, W)`` plane.
    n_blocks : int, optional
        Number of blocks on the short side, by default
        :data:`~aind_flatfield_correction.core.basicpy.config.BG_GRID_BLOCKS`.

    Returns
    -------
    np.ndarray
        The background map, one value per block.
    """
    plane = np.asarray(plane, dtype=np.float32)
    block = max(1, min(plane.shape) // n_blocks)
    rows, cols = plane.shape[0] // block, plane.shape[1] // block
    blocks = (
        plane[: rows * block, : cols * block]
        .reshape(rows, block, cols, block)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols, -1)
    )
    return gaussian(
        np.median(blocks, axis=2),
        sigma=1.0,
        mode="reflect",
        preserve_range=True,
    )


def plane_scores(fit_slices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Score every plane for brightness and for concentrated signal.

    ``localized`` is the 99th percentile of the plane's background map over
    its median.  Diffuse signal gives a value near the range of the
    illumination itself (about 1.2-2.5); one region much brighter than the
    rest of the plane gives a much larger value (median 46 on SmartSPIM 561,
    where the flatfield learned the tissue).  A plane whose median is not
    above zero is mostly empty, its ratio is meaningless and is NaN, so such
    a plane never counts as concentrated.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.

    Returns
    -------
    dict
        ``mean`` (plane mean), ``level`` (median of the background map) and
        ``localized`` (99th percentile over median), one value per plane.
    """
    n_planes = fit_slices.shape[0]
    means = np.asarray(fit_slices, dtype=np.float64).mean(axis=(1, 2))
    level = np.zeros(n_planes, dtype=np.float64)
    localized = np.full(n_planes, np.nan, dtype=np.float64)
    for index in range(n_planes):
        bg_map = background_map(fit_slices[index])
        level[index] = float(np.median(bg_map))
        if level[index] > 0:
            localized[index] = float(np.percentile(bg_map, 99)) / level[index]
    return {"mean": means, "level": level, "localized": localized}


def concentrated_planes(
    scores: dict[str, np.ndarray],
    bright_ratio: float = BRIGHT_RATIO,
    localized_max: float = LOCALIZED_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the planes that are bright and have one dominant bright region.

    Both conditions are needed.  Bright planes whose tissue covers the whole
    view are among the best planes for a flatfield, and a concentrated plane
    that is not bright carries little of the fit objective.

    Parameters
    ----------
    scores : dict
        Output of :func:`plane_scores`.
    bright_ratio : float, optional
        Multiple of the median plane mean above which a plane is bright.
    localized_max : float, optional
        ``localized`` value above which a plane is concentrated.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Boolean masks of the bright planes and of the planes that are both
        bright and concentrated.
    """
    means = scores["mean"]
    median_mean = float(np.median(means))
    if median_mean > 0:
        bright = means > bright_ratio * median_mean
    else:
        bright = np.zeros(means.shape, dtype=bool)
    # NaN (a mostly empty plane) compares False, so it is never dropped.
    with np.errstate(invalid="ignore"):
        concentrated = scores["localized"] > localized_max
    return bright, bright & concentrated


def drop_concentrated_planes(
    fit_slices: np.ndarray,
    stack: FitStack,
    scores: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, FitStack, dict]:
    """
    Remove the bright, concentrated planes before the joint fit.

    BaSiC sorts the values of each pixel across the images, so its top
    sorted images are the brightest value any plane has at each pixel.  When
    the bright planes hold one dominant region each, those top images show
    where the regions are, and the brightness-weighted flatfield update
    follows them: the flatfield learns the tissue.  Removing these planes
    fixed SmartSPIM 561 on both laser sides and left the flatfields of the
    other channels unchanged.

    The per-tile spans and the tile manifest are rebuilt for the kept
    planes.  A tile can be left with no planes; the search's subsampling
    skips an empty span.  At least half of the planes are always kept: a
    plane is bright only above a multiple of the median plane mean.

    Parameters
    ----------
    fit_slices : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.
    stack : FitStack
        The raw stack the fitting stack was made from.
    scores : dict or None, optional
        Output of :func:`plane_scores`, computed here when None.

    Returns
    -------
    tuple of (np.ndarray, FitStack, dict)
        The kept fitting planes, the stack rebuilt for them, and a report of
        what was dropped.
    """
    if scores is None:
        scores = plane_scores(fit_slices)
    _, drop = concentrated_planes(scores)
    report = {
        "enabled": True,
        "rule": {
            "bright_ratio": BRIGHT_RATIO,
            "localized_max": LOCALIZED_MAX,
        },
        "n_candidates": int(drop.size),
        "top_share_before": plane_concentration(fit_slices)["top_share"],
    }
    keep = ~drop

    z_offsets: dict[int, tuple[int, int]] = {}
    manifest: list[TileRecord] = []
    dropped_per_tile: dict[str, int] = {}
    offset = 0
    for index in sorted(stack.z_offsets):
        start, end = stack.z_offsets[index]
        tile_keep = keep[start:end]
        n_kept = int(tile_keep.sum())
        z_offsets[index] = (offset, offset + n_kept)
        offset += n_kept
        if index < len(stack.manifest):
            record = stack.manifest[index]
            z_kept = tuple(
                z for z, kept in zip(record.z_indices, tile_keep) if kept
            )
            manifest.append(record._replace(z_indices=z_kept))
            if n_kept < end - start:
                dropped_per_tile[record.name] = end - start - n_kept

    kept_slices = fit_slices[keep]
    new_stack = stack._replace(
        slices=stack.slices[keep],
        z_offsets=z_offsets,
        manifest=tuple(manifest),
    )
    report.update(
        {
            "n_dropped": int(drop.sum()),
            "n_kept": int(keep.sum()),
            "dropped_per_tile": dropped_per_tile,
            "top_share_after": plane_concentration(kept_slices)["top_share"],
        }
    )
    logger.info(
        "  Plane filter: dropped %d of %d planes (bright and concentrated); "
        "top-plane share %.3f -> %.3f",
        report["n_dropped"],
        report["n_candidates"],
        report["top_share_before"],
        report["top_share_after"],
    )
    return kept_slices, new_stack, report
