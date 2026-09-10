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
from typing import NamedTuple

import boto3
import numpy as np
from aind_large_scale_prediction.io import OMEZarrReader
from botocore import UNSIGNED
from botocore.client import Config
from skimage.transform import resize as sk_resize
from tifffile import imread as tif_imread

logger = logging.getLogger(__name__)


class FitStack(NamedTuple):
    """
    Raw planes gathered from every tile of one channel.

    ``z_offsets`` maps a tile's index in the tile-name list to its
    ``(start, end)`` span in ``slices``.
    """

    slices: np.ndarray
    z_offsets: dict[int, tuple[int, int]]
    n_planes: int
    height: int
    width: int


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


def open_tile(base_path: str, name: str, level: int | str):
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

    Returns
    -------
    dask.array.Array
        Lazy ``(Z, Y, X)`` view of the tile; leading singleton axes are
        dropped.
    """
    uri = f"{base_path.rstrip('/')}/{name}"
    reader = OMEZarrReader(
        data_path=uri, multiscale=str(level), zarr_version="3.0"
    )
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
        per tile, and the padded plane height and width.
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
    return FitStack(stack, z_offsets, n_planes, max_h, max_w)


def load_darkfield(
    darkfield_image: str | None, darkfield_value: float
) -> np.ndarray | float:
    """
    Resolve the darkfield camera offset from the command-line arguments.

    Parameters
    ----------
    darkfield_image : str or None
        Path to a ``.npy`` darkfield image. Takes precedence.
    darkfield_value : float
        Scalar pedestal in ADU counts, used when no image is given.

    Returns
    -------
    np.ndarray or float
        A 2-D/3-D darkfield image, or the scalar pedestal.
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

        logger.info(
            "  Darkfield image %s: shape=%s mean=%.2f",
            darkfield_image,
            dark.shape,
            float(dark.mean()),
        )
        return dark
    return float(darkfield_value)


def match_darkfield(
    dark: np.ndarray | float, shape: tuple[int, int]
) -> np.ndarray:
    """
    Broadcast or resize the darkfield onto a ``(height, width)`` plane.

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
    # Resizing is necessary since we could have a high resolution
    # darkfield image but using a downsampled version for the estimation
    if dark.shape != tuple(shape):
        logger.info(
            "  Darkfield resized %s -> %s for the estimation level",
            dark.shape,
            tuple(shape),
        )
        dark = sk_resize(
            dark,
            shape,
            order=1,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        )
    return np.asarray(dark, dtype=np.float32)


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
