"""
Inspection figures for a fitted flatfield.

The figures are the acceptance gate: the X/Y intensity profiles show the
raw curve bowed by vignetting and the corrected curve flattened, so a
flatfield can be reviewed before it is applied to a whole dataset.

Nothing here needs a tile's position on the acquisition grid.  Every
figure is built either from a single tile's own planes or from the mean
of the whole fitting stack, so no stitched composite is assembled.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from aind_flatfield_correction.core.basicpy.config import (
    SIGNAL_LOW_FRACTION,
    Z_FRACTION,
)
from aind_flatfield_correction.core.basicpy.tiles import FitStack
from aind_flatfield_correction.metrics.metrics import (
    cv,
    masked_profile,
    signal_cv,
)

logger = logging.getLogger(__name__)


def _get_pyplot():
    """
    Import pyplot with a headless backend.

    Imported lazily so that estimation runs that do not ask for figures
    never pay for matplotlib, and so the backend is fixed before the
    first figure is created.

    Returns
    -------
    module
        ``matplotlib.pyplot``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def correct_slice(
    slc: np.ndarray, flatfield: np.ndarray, dark_plane: np.ndarray
) -> np.ndarray:
    """
    Apply the correction to a single plane, for visualization.

    Parameters
    ----------
    slc : np.ndarray
        Raw ``(H, W)`` plane.
    flatfield : np.ndarray
        Fitted flatfield, broadcastable to ``slc``.
    dark_plane : np.ndarray
        Darkfield plane removed before dividing.

    Returns
    -------
    np.ndarray
        Corrected plane, float32.
    """
    out = np.clip(slc.astype(np.float32) - dark_plane, 0.0, None)
    flat = np.where(flatfield <= 0, 1e-6, flatfield)
    return out / flat


def pick_sample_tiles(n_tiles: int, n_wanted: int) -> list[int]:
    """
    Pick evenly spaced tile indices to plot profiles for.

    Tile names carry no grid position, so the sample cannot be aimed at
    the mosaic extremes.  Spreading the picks over the (sorted) tile list
    is the next best thing: acquisition order tends to sweep the mosaic,
    so an even spread still lands on tiles from different regions.

    Parameters
    ----------
    n_tiles : int
        Number of tiles available.
    n_wanted : int
        Number of tiles wanted.

    Returns
    -------
    list of int
        Sorted, unique indices into the tile list.
    """
    if n_tiles <= 0 or n_wanted <= 0:
        return []
    if n_wanted >= n_tiles:
        return list(range(n_tiles))
    picks = np.linspace(0, n_tiles - 1, n_wanted)
    return sorted({int(round(pick)) for pick in picks})


def _safe_name(name: str) -> str:
    """
    Reduce a tile name to something usable in a filename.

    Parameters
    ----------
    name : str
        Tile directory name.

    Returns
    -------
    str
        The name without its OME-Zarr suffix and with every run of
        non-alphanumeric characters collapsed to a single underscore.
    """
    stem = re.sub(r"\.ome\.zarr/?$", "", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def _display_range(image: np.ndarray) -> tuple[float, float]:
    """
    Robust display limits over the occupied pixels of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to scale.

    Returns
    -------
    tuple of (float, float)
        The 0.5 and 99.5 percentiles of the positive pixels.
    """
    values = image[image > 0]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, (0.5, 99.5))
    return float(low), float(high)


def _display_plane(stack: FitStack, index: int) -> np.ndarray:
    """
    Return the display plane of one tile from the fitting stack.

    Parameters
    ----------
    stack : FitStack
        Loaded planes and their per-tile spans.
    index : int
        Index of the tile in the tile-name list.

    Returns
    -------
    np.ndarray
        The raw ``(H, W)`` plane at :data:`Z_FRACTION` through the
        tile's span.
    """
    start, end = stack.z_offsets[index]
    offset = start + int(round(Z_FRACTION * (end - start - 1)))
    return stack.slices[offset]


def _plot_profile_pair(
    axis,
    before: np.ndarray,
    after: np.ndarray,
    name: str,
) -> None:
    """
    Draw one before/after intensity profile onto an axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to draw on.
    before : np.ndarray
        Pedestal-subtracted raw profile.
    after : np.ndarray
        Corrected profile.
    name : str
        Axis label, ``"X"`` or ``"Y"``.

    Returns
    -------
    None
    """
    axis.plot(
        before,
        label="Raw - darkfield",
        color="steelblue",
        linewidth=1.2,
    )
    axis.plot(after, label="Corrected", color="tomato", linewidth=1.2)
    axis.set_xlabel(f"{name} pixel")
    axis.set_ylabel("Mean intensity")
    axis.set_title(f"{name} intensity profile")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)


def signal_change_profile(
    before: np.ndarray, after: np.ndarray
) -> np.ndarray:
    """
    Percent change the correction applies at each position of a profile.

    The ratio of the corrected profile to the pedestal-subtracted raw one
    is the gain averaged over that row or column, weighted by the tile's
    own signal, so it shows where the correction brightens or dims the
    data and by how much.

    Parameters
    ----------
    before : np.ndarray
        Pedestal-subtracted raw profile.
    after : np.ndarray
        Corrected profile.

    Returns
    -------
    np.ndarray
        ``100 * (after / before - 1)``, NaN wherever ``before`` holds no
        signal.
    """
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    ratio = np.divide(
        after, before, out=np.full(before.shape, np.nan), where=before > 0
    )
    return 100.0 * (ratio - 1.0)


def _plot_change_profile(
    axis, raw_profile: np.ndarray, corrected: np.ndarray
) -> None:
    """
    Draw the percent change in signal by position, along both axes.

    Both curves share an edge-centre-edge axis, and signal lost is shaded
    red: a mean-1 flatfield dims the bright centre of the tile, and this
    panel is where that shows up.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to draw on.
    raw_profile : np.ndarray
        Pedestal-subtracted raw plane.
    corrected : np.ndarray
        Corrected plane.

    Returns
    -------
    None
    """
    for along, name, colour in (
        (1, "top → bottom", "#2a78d6"),
        (0, "left → right", "#eb6834"),
    ):
        change = signal_change_profile(
            raw_profile.mean(axis=along), corrected.mean(axis=along)
        )
        finite = np.isfinite(change)
        if finite.any():
            label = (
                f"{name}: {change[finite].min():+.0f}% to "
                f"{change[finite].max():+.0f}%"
            )
        else:
            label = f"{name}: no signal above the darkfield"
        position = np.linspace(0.0, 1.0, change.size)
        axis.plot(position, change, color=colour, linewidth=2.0, label=label)
        axis.fill_between(
            position,
            change,
            0.0,
            where=np.nan_to_num(change) < 0,
            color="#d03b3b",
            alpha=0.25,
            interpolate=True,
        )
    low, high = axis.get_ylim()
    axis.set_ylim(min(low, -5.0), max(high, 5.0))
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks([0.0, 0.5, 1.0], ["edge", "centre", "edge"])
    axis.set_ylabel("Change in signal (%)")
    axis.set_title("Change in signal by position (red = lost)")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)


def _plot_overview(
    label: str,
    flatfield: np.ndarray,
    plane: np.ndarray,
    corrected: np.ndarray,
    level: int | str,
    out_path: Path,
) -> None:
    """
    Save the flatfield beside one example plane, raw and corrected.

    Parameters
    ----------
    label : str
        Run label, for the title.
    flatfield : np.ndarray
        Fitted flatfield.
    plane : np.ndarray
        Raw example plane.
    corrected : np.ndarray
        The same plane, corrected.
    level : int or str
        Pyramid level the estimation ran on.
    out_path : Path
        PNG path to write.

    Returns
    -------
    None
    """
    plt = _get_pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))

    image = axs[0].imshow(flatfield, cmap="inferno")
    axs[0].set_title(f"Flatfield (level {level})", fontsize=11)
    plt.colorbar(image, ax=axs[0], fraction=0.046)
    axs[0].axis("off")

    for axis, panel, title in (
        (axs[1], plane, "Example plane, raw"),
        (axs[2], corrected, "Example plane, corrected"),
    ):
        vmin, vmax = _display_range(panel)
        axis.imshow(
            panel,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    fig.suptitle(f"{label} - BaSiC flatfield overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tile_profile(
    label: str,
    tile_name: str,
    plane: np.ndarray,
    corrected: np.ndarray,
    dark_plane: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save the raw / X profile / Y profile / corrected / change panels.

    The profile curves compare pedestal-subtracted raw against corrected
    values: the correction removes the offset, so plotting true raw
    against it would shift the curves apart and hide whether the
    vignetting was actually flattened.  The last panel is the ratio of
    those two curves, as a percent change by position, so a correction
    that dims part of the tile shows up as a red region.

    Parameters
    ----------
    label : str
        Run label, for the title.
    tile_name : str
        Name of the tile being plotted.
    plane : np.ndarray
        Raw display plane.
    corrected : np.ndarray
        Corrected display plane.
    dark_plane : np.ndarray
        Darkfield plane.
    out_path : Path
        PNG path to write.

    Returns
    -------
    None
    """
    plt = _get_pyplot()
    raw_profile = np.clip(plane - dark_plane, 0.0, None)
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))
    fig.suptitle(f"{label}  {tile_name}", fontsize=11)

    vmin, vmax = _display_range(plane)
    axs[0].imshow(
        plane,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axs[0].set_title("Raw")
    axs[0].axis("off")

    _plot_profile_pair(
        axs[1], raw_profile.mean(axis=0), corrected.mean(axis=0), "X"
    )
    _plot_profile_pair(
        axs[2], raw_profile.mean(axis=1), corrected.mean(axis=1), "Y"
    )

    vmin, vmax = _display_range(corrected)
    axs[3].imshow(
        corrected,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axs[3].set_title("Corrected")
    axs[3].axis("off")

    _plot_change_profile(axs[4], raw_profile, corrected)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def low_plane_mean(
    slices: np.ndarray, fraction: float = SIGNAL_LOW_FRACTION
) -> np.ndarray:
    """
    Average the dimmest planes of a stack.

    Parameters
    ----------
    slices : np.ndarray
        ``(N, H, W)`` stack of raw planes.
    fraction : float, optional
        Share of the planes to average, the dimmest by plane mean; at
        least one plane is used.

    Returns
    -------
    np.ndarray
        The ``(H, W)`` mean of the dimmest planes.
    """
    plane_means = np.asarray(slices, dtype=np.float64).mean(axis=(1, 2))
    n_low = max(1, int(round(fraction * plane_means.size)))
    dimmest = np.argsort(plane_means)[:n_low]
    return np.asarray(slices[np.sort(dimmest)], dtype=np.float64).mean(axis=0)


def _plot_vignetting(
    label: str,
    mean_raw: np.ndarray,
    flatfield: np.ndarray,
    dark_plane: np.ndarray,
    out_path: Path,
    mean_low: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Profile the mean of every fitted plane, before and after correction.

    Averaging the whole fitting stack rather than one plane per tile
    keeps the vignetting signal and averages the specimen away, and it
    needs no tile positions.  The profiles average occupied pixels only:
    tiles smaller than the padded plane leave zeros behind, and
    averaging those in as real values would flatten the curves.

    The ``cv_*`` values divide by the mean of the corrected image, so a
    darkfield that is too low makes them look better.  When ``mean_low``
    is given, the ``signal_cv_*`` values are also returned: the same
    profiles of the mean minus ``mean_low``, in which the darkfield
    cancels (:func:`~aind_flatfield_correction.metrics.metrics.signal_cv`).

    Parameters
    ----------
    label : str
        Run label, for the title.
    mean_raw : np.ndarray
        Mean of every raw plane in the fitting stack.
    flatfield : np.ndarray
        Fitted flatfield.
    dark_plane : np.ndarray
        Darkfield plane.
    out_path : Path
        PNG path to write.
    mean_low : np.ndarray or None, optional
        Mean of the dimmest raw planes (:func:`low_plane_mean`).

    Returns
    -------
    dict
        Coefficients of variation before and after correction, per axis.
    """
    plt = _get_pyplot()
    before = np.clip(mean_raw - dark_plane, 0.0, None)
    after = correct_slice(mean_raw, flatfield, dark_plane)
    profiles = {
        "x": (masked_profile(before, 0), masked_profile(after, 0)),
        "y": (masked_profile(before, 1), masked_profile(after, 1)),
    }

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for axis, key in ((axs[0], "x"), (axs[1], "y")):
        _plot_profile_pair(axis, *profiles[key], key.upper())
        axis.set_title(f"{key.upper()}-axis vignetting profile")

    fig.suptitle(
        f"{label} - vignetting profiles, mean of all fitted planes",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {}
    for key, (raw_profile, corrected) in profiles.items():
        metrics[f"cv_{key}_before"] = cv(raw_profile)
        metrics[f"cv_{key}_after"] = cv(corrected)
        logger.info(
            "  CV - %s: raw-darkfield=%.2f%%  corrected=%.2f%%",
            key.upper(),
            metrics[f"cv_{key}_before"],
            metrics[f"cv_{key}_after"],
        )
    if mean_low is not None:
        for axis, key in ((0, "x"), (1, "y")):
            metrics[f"signal_cv_{key}_before"] = signal_cv(
                mean_raw, mean_low, 1.0, axis
            )
            metrics[f"signal_cv_{key}_after"] = signal_cv(
                mean_raw, mean_low, flatfield, axis
            )
            logger.info(
                "  Signal CV - %s: before=%.2f%%  corrected=%.2f%%",
                key.upper(),
                metrics[f"signal_cv_{key}_before"],
                metrics[f"signal_cv_{key}_after"],
            )
    return metrics


def save_validation_figures(
    label: str,
    flatfield: np.ndarray,
    dark_plane: np.ndarray,
    stack: FitStack,
    tile_names: list[str],
    output_dir: Path,
    level: int | str,
    n_profile_tiles: int,
) -> dict[str, Any]:
    """
    Write every inspection figure for one run.

    Parameters
    ----------
    label : str
        Run label, used in titles and filenames.
    flatfield : np.ndarray
        Fitted flatfield.
    dark_plane : np.ndarray
        Darkfield plane used for the fit.
    stack : FitStack
        Planes the fit was built from.
    tile_names : list of str
        Tiles that contributed to the fit, in stack order.
    output_dir : Path
        Folder to write the PNGs into; created if missing.
    level : int or str
        Pyramid level the estimation ran on.
    n_profile_tiles : int
        Number of tiles to write per-tile profile figures for.

    Returns
    -------
    dict
        Coefficients of variation plus the figure paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = pick_sample_tiles(len(tile_names), n_profile_tiles)

    vignetting = output_dir / f"{label}_vignetting_profiles.png"
    metrics: dict[str, Any] = _plot_vignetting(
        label,
        stack.slices.mean(axis=0),
        flatfield,
        dark_plane,
        vignetting,
        low_plane_mean(stack.slices),
    )
    written = [str(vignetting)]

    logger.info(
        "  Writing X/Y profile plots for %d of %d tiles",
        len(indices),
        len(tile_names),
    )
    for index in indices:
        plane = _display_plane(stack, index)
        corrected = correct_slice(plane, flatfield, dark_plane)
        name = tile_names[index]
        path = output_dir / f"{label}_tile{index:03d}_{_safe_name(name)}.png"
        _plot_tile_profile(label, name, plane, corrected, dark_plane, path)
        written.append(str(path))

        if index == indices[0]:
            overview = output_dir / f"{label}_overview.png"
            _plot_overview(label, flatfield, plane, corrected, level, overview)
            written.append(str(overview))

    metrics["figures"] = written
    logger.info("  Figures -> %s", output_dir)
    return metrics
