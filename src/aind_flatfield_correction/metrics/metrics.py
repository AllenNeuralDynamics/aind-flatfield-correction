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
