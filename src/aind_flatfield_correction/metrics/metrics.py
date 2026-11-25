"""
Metrics for evaluating flatfield correction.
"""

import numpy as np


def cv(arr: np.ndarray) -> float:
    """
    Coefficient of variation (%), ignoring NaNs.

    Parameters
    ----------
    arr : array-like
        Input array.

    Returns
    -------
    float
        Coefficient of variation (%) of the input array, ignoring NaNs.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    mean = np.nanmean(arr)
    return float(np.nanstd(arr) / mean * 100) if mean else float("nan")


def masked_profile(mosaic, axis):
    """
    Mean along ``axis`` over occupied pixels only.

    Parameters
    ----------
    mosaic : array-like
        Input tile mosaic.
    axis : int
        Axis along which to compute the mean.

    Returns
    -------
    np.ndarray
        Mean profile along the specified axis, ignoring empty grid cells.
    """
    m = np.where(np.asarray(mosaic) > 0, mosaic, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(m, axis=axis)


def signal_cv(mean_all, mean_low, flatfield, axis):
    """
    Coefficient of variation (%) of the signal profile after correction.

    The signal is the mean of all the planes minus the mean of the dimmest
    planes.  The darkfield is in both means, so it cancels exactly: unlike
    :func:`cv` of a corrected image, this does not look better when a
    correction leaves part of the darkfield in the image (a larger mean
    makes any spread look small).  It therefore tests the flatfield alone.

    Parameters
    ----------
    mean_all : array-like
        Mean of all the planes, raw or with the darkfield removed.
    mean_low : array-like
        Mean of the dimmest planes, in the same units as ``mean_all``.
    flatfield : array-like or float
        Flatfield to correct with; 1.0 gives the value before correction.
    axis : int
        Axis along which the profile is averaged.

    Returns
    -------
    float
        Coefficient of variation (%) of the corrected signal profile.
    """
    signal = np.asarray(mean_all, dtype=np.float64) - np.asarray(
        mean_low, dtype=np.float64
    )
    return cv(masked_profile(signal / flatfield, axis))
