"""
Parameter search for the BaSiC flatfield fit.

basicpy's autotune objective is ``return -entropy`` handed to a
maximizing optimizer, i.e. it MINIMIZES entropy.  Everything here does
argmin: an earlier version that did argmax on raw entropy selected the
flattest possible flatfield, because leaving vignetting in place keeps
the intensity histogram spread out and so scores high.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from typing import Any, NamedTuple

import numpy as np
from basicpy import BaSiC

from aind_flatfield_correction.core.basicpy.config import (
    MANUAL_PARAMS,
    SEARCH_GRID,
    SELECT_TOL,
    limit_worker_threads,
    worker_pool_size,
)

logger = logging.getLogger(__name__)


class Baseline(NamedTuple):
    """Reference fit with ``MANUAL_PARAMS``, used as the incumbent."""

    val_range: float
    entropy: float
    std: float
    fit_seconds: float


def basic_entropy(
    transformed: np.ndarray,
    vmin: float,
    vmax: float,
    bins: int = 100,
) -> float:
    """
    Replicate ``basicpy.metrics.entropy`` in float64.

    Parameters
    ----------
    transformed : np.ndarray
        Output of ``BaSiC.transform``.
    vmin : float
        Lower edge of the histogram window.
    vmax : float
        Upper edge of the histogram window.
    bins : int, optional
        Histogram bin count, by default 100.

    Returns
    -------
    float
        Differential entropy in nats, or ``+inf`` when it cannot be
        computed.
    """
    values = np.asarray(transformed, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    values = values[(values >= vmin) & (values <= vmax)]  # DROP, not clip
    if values.size == 0:
        return float("inf")
    prob_density, edges = np.histogram(
        values, bins=bins, range=(float(vmin), float(vmax)), density=True
    )
    width = float(edges[1] - edges[0])
    prob_density = prob_density[prob_density > 0]
    if prob_density.size == 0:
        return float("inf")
    entropy = float(-np.sum(prob_density * np.log(prob_density)) * width)
    return entropy if np.isfinite(entropy) else float("inf")


def score_fit(
    basic: BaSiC,
    images: np.ndarray,
    val_range: float,
    bins: int = 100,
) -> float:
    """
    Entropy of a fitted BaSiC's transform of ``images``. lower is better.

    Parameters
    ----------
    basic : basicpy.BaSiC
        Already-fitted model.
    images : np.ndarray
        Images to transform and score.
    val_range : float
        Frozen histogram window width.
    bins : int, optional
        Histogram bin count, by default 100.

    Returns
    -------
    float
        Entropy of the transformed images.
    """
    transformed = np.asarray(basic.transform(images), dtype=np.float64)
    vmin_new = float(np.quantile(transformed, 0.01))
    return basic_entropy(
        transformed, vmin_new, vmin_new + val_range, bins=bins
    )


def stratified_subsample(
    images: np.ndarray,
    n_target: int,
    tile_z_offsets: dict[int, tuple[int, int]] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    Take about ``n_target`` slices, drawing equally from every tile.

    The flatfield is shared across all tiles, so every field position
    must be represented equally.  A uniform draw over the concatenated Z
    axis over-weights whichever tiles happen to be sampled more and
    distorts the fit.

    Parameters
    ----------
    images : np.ndarray
        Concatenated ``(N, H, W)`` stack.
    n_target : int
        Desired number of slices.
    tile_z_offsets : dict, optional
        Per-tile ``(start, end)`` spans into ``images``. Without it the
        draw is uniform over the whole stack.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    tuple of (np.ndarray, int)
        The subsampled stack and its slice count.
    """
    n_total = images.shape[0]
    if n_total <= n_target:
        return images, n_total
    # Pick a random range
    rng = np.random.default_rng(seed)
    if tile_z_offsets:
        per_tile = max(1, math.ceil(n_target / len(tile_z_offsets)))
        picks = []
        for start, end in sorted(tile_z_offsets.values()):
            span = end - start
            if span <= 0:
                continue
            n_pick = min(per_tile, span)
            picks.append(start + rng.choice(span, n_pick, replace=False))
        idx = np.sort(np.concatenate(picks))
        if idx.size > n_target:  # trim evenly, keep tile coverage
            keep = np.linspace(0, idx.size - 1, n_target)
            idx = idx[keep.round().astype(int)]
    else:
        idx = np.sort(rng.choice(n_total, n_target, replace=False))
    return np.ascontiguousarray(images[idx]), int(idx.size)


def _eval_basic_params(
    shm_name: str,
    images_shape: tuple[int, ...],
    images_dtype: np.dtype,
    params: dict[str, float],
    base_config: dict[str, Any],
    val_range: float,
    screen_iters: int | None = None,
    threads_per_worker: int = 2,
) -> tuple[float, dict[str, float], str | None]:
    """
    Worker: attach to the shared stack, fit BaSiC, score the fit.

    Parameters
    ----------
    shm_name : str
        Name of the shared-memory block holding the images.
    images_shape : tuple of int
        Shape of the shared image array.
    images_dtype : np.dtype
        Dtype of the shared image array.
    params : dict
        Candidate BaSiC parameters to overlay on ``base_config``.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    val_range : float
        Frozen histogram window width for scoring.
    screen_iters : int or None, optional
        Reduced ``max_reweight_iterations`` for the cheap screen. None
        keeps the full-fidelity value.
    threads_per_worker : int, optional
        BLAS thread cap, by default 2.

    Returns
    -------
    tuple of (float, dict, str or None)
        Entropy (``+inf`` on failure), the candidate parameters, and the
        traceback (None on success).
    """
    limit_worker_threads(threads_per_worker)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Putting the images on a shared memory buffer
        images = np.ndarray(images_shape, dtype=images_dtype, buffer=shm.buf)
        config = dict(base_config)
        if screen_iters is not None:  # stage-1 screen only; never decides
            config["max_reweight_iterations"] = screen_iters
        basic = BaSiC(**{**config, **params})
        basic.fit(images)
        return float(score_fit(basic, images, val_range)), params, None
    except Exception:  # noqa: BLE001 - reported to the parent
        import traceback

        return float("inf"), params, traceback.format_exc()
    finally:
        # Always close the shared memory handle
        shm.close()


def _run_candidate_wave(
    eval_images: np.ndarray,
    candidates: list[float],
    base_config: dict[str, Any],
    val_range: float,
    n_workers: int,
    screen_iters: int | None,
    threads_per_worker: int,
    label: str,
) -> dict[float, float]:
    """
    Score a batch of candidates in parallel.

    Shares the image stack through POSIX shared memory so spawned
    workers attach to it instead of pickling a copy.

    Parameters
    ----------
    eval_images : np.ndarray
        Images every candidate is fitted and scored on.
    candidates : list of float
        ``smoothness_flatfield`` values to score.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    val_range : float
        Frozen histogram window width for scoring.
    n_workers : int
        Worker processes to run.
    screen_iters : int or None
        Reduced ``max_reweight_iterations`` for the cheap screen.
    threads_per_worker : int
        BLAS thread cap per worker.
    label : str
        Short tag used in the progress log.

    Returns
    -------
    dict
        ``{smoothness_flatfield: entropy}``.
    """
    scores: dict[float, float] = {}
    shm = shared_memory.SharedMemory(create=True, size=eval_images.nbytes)
    try:
        view = np.ndarray(
            eval_images.shape, dtype=eval_images.dtype, buffer=shm.buf
        )
        view[:] = eval_images
        mp_ctx = multiprocessing.get_context("spawn")
        # Submitting the round of candidate evaluations to the process pool
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_ctx
        ) as pool:
            futures = {
                pool.submit(
                    _eval_basic_params,
                    shm.name,
                    eval_images.shape,
                    eval_images.dtype,
                    {"smoothness_flatfield": smoothness},
                    base_config,
                    val_range,
                    screen_iters,
                    threads_per_worker,
                ): smoothness
                for smoothness in candidates
            }
            done = 0
            for future in as_completed(futures):
                entropy, params, err = future.result()
                smoothness = params["smoothness_flatfield"]
                scores[smoothness] = entropy
                done += 1
                if err:
                    logger.error(
                        "    [%s] smoothness_flatfield=%g FAILED:\n%s",
                        label,
                        smoothness,
                        err,
                    )
                logger.info(
                    "    [%s] %d/%d  smoothness_flatfield=%-10.4g "
                    "entropy=%.6f",
                    label,
                    done,
                    len(futures),
                    smoothness,
                    entropy,
                )
    finally:
        # Always close and unlink the shared memory handle
        shm.close()
        shm.unlink()
    return scores


def _fit_baseline(
    eval_images: np.ndarray, base_config: dict[str, Any]
) -> Baseline:
    """
    Fit the reference model that anchors the whole search.

    Parameters
    ----------
    eval_images : np.ndarray
        Images the search will score on.
    base_config : dict
        BaSiC keyword arguments, used verbatim.

    Returns
    -------
    Baseline
        Frozen window width, baseline entropy, flatfield std, and the
        wall-clock cost of one full-fidelity fit.
    """
    started = time.monotonic()
    ref = BaSiC(**{**base_config, **MANUAL_PARAMS})
    ref.fit(eval_images)
    transformed = np.asarray(ref.transform(eval_images), dtype=np.float64)
    vmin, vmax = np.quantile(transformed, [0.01, 0.99])
    val_range = float(vmax - vmin)  # FROZEN for the whole search
    entropy = basic_entropy(transformed, float(vmin), float(vmin) + val_range)
    elapsed = time.monotonic() - started
    flat = np.asarray(ref.flatfield)
    logger.info(
        "  Baseline (manual) smoothness_flatfield=%g: entropy=%.6f  "
        "flatfield std=%.4f span=%.4f",
        MANUAL_PARAMS["smoothness_flatfield"],
        entropy,
        float(flat.std()),
        float(np.ptp(flat)),
    )
    return Baseline(val_range, entropy, float(flat.std()), elapsed)


def _project_search_minutes(
    baseline: Baseline,
    base_config: dict[str, Any],
    n_workers: int,
    screen_iters: int,
    n_stage2: int,
) -> float:
    """
    Project the optimistic wall-clock cost of the two-stage search.

    Parameters
    ----------
    baseline : Baseline
        Carries the measured cost of one full-fidelity fit.
    base_config : dict
        BaSiC keyword arguments, for the full iteration count.
    n_workers : int
        Worker processes available.
    screen_iters : int
        Reduced iteration count used by the stage-1 screen.
    n_stage2 : int
        Finalists re-scored at full fidelity.

    Returns
    -------
    float
        Optimistic projection in minutes.
    """
    full_iters = base_config.get("max_reweight_iterations", 10)
    waves_1 = math.ceil(len(SEARCH_GRID) / n_workers)
    waves_2 = math.ceil((n_stage2 + 1) / n_workers)
    projected = (
        baseline.fit_seconds
        * (waves_1 * (screen_iters / full_iters) + waves_2)
        / 60.0
    )
    logger.info(
        "  Calibration: one full-config fit = %.0fs  |  projected "
        "search %.0f-%.0f min (stage1 %d cands x%d iters, stage2 %d "
        "cands x%d iters)",
        baseline.fit_seconds,
        projected,
        3 * projected,
        len(SEARCH_GRID),
        screen_iters,
        n_stage2 + 1,
        full_iters,
    )
    return projected


def _select_winner(
    scores: dict[float, float], baseline_entropy: float
) -> tuple[dict[str, float], float]:
    """
    Pick the lowest-entropy candidate, with the baseline as incumbent.

    Parameters
    ----------
    scores : dict
        ``{smoothness_flatfield: entropy}`` at full fidelity.
    baseline_entropy : float
        Entropy of the manual parameters.

    Returns
    -------
    tuple of (dict, float)
        Winning parameters and their entropy.
    """
    best_params, best_entropy = dict(MANUAL_PARAMS), baseline_entropy
    for smoothness in sorted(scores):  # deterministic tie ordering
        if scores[smoothness] < best_entropy - SELECT_TOL:
            best_params = {"smoothness_flatfield": smoothness}
            best_entropy = scores[smoothness]
    return best_params, best_entropy


def _log_ranking(
    scores: dict[float, float],
    baseline_smoothness: float,
    best_params: dict[str, float],
) -> None:
    """
    Log the stage-2 ranking, flagging the baseline and the winner.

    Parameters
    ----------
    scores : dict
        ``{smoothness_flatfield: entropy}`` at full fidelity.
    baseline_smoothness : float
        The incumbent's ``smoothness_flatfield``.
    best_params : dict
        The selected parameters.

    Returns
    -------
    None
    """
    logger.info(
        "  Ranked stage-2 candidates (LOWER entropy = better " "correction):"
    )
    for smoothness, entropy in sorted(
        scores.items(), key=lambda item: item[1]
    ):
        tags = []
        if smoothness == baseline_smoothness:
            tags.append("BASELINE")
        if smoothness == best_params["smoothness_flatfield"]:
            tags.append("CHOSEN")
        suffix = ("   <-- " + ", ".join(tags)) if tags else ""
        logger.info(
            "    smoothness_flatfield=%-10.4g entropy=%.6f%s",
            smoothness,
            entropy,
            suffix,
        )


def _screen_and_rescore(
    eval_images: np.ndarray,
    base_config: dict[str, Any],
    baseline: Baseline,
    n_workers: int,
    threads: int,
    screen_iters: int,
    n_stage2: int,
) -> dict[float, float] | None:
    """
    Screen the whole grid cheaply, then re-score the finalists.

    Stage 1 ranks every grid point at a reduced iteration count and
    never decides anything; stage 2 re-scores its best few, plus the
    baseline, under the exact solver the flatfield will be fitted with.

    Parameters
    ----------
    eval_images : np.ndarray
        Images every candidate is fitted and scored on.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    baseline : Baseline
        The incumbent, whose frozen window and entropy are reused.
    n_workers : int
        Worker processes to run.
    threads : int
        BLAS thread cap per worker.
    screen_iters : int
        Reduced iteration count for the stage-1 screen.
    n_stage2 : int
        Finalists re-scored at full fidelity.

    Returns
    -------
    dict or None
        ``{smoothness_flatfield: entropy}`` at full fidelity, or None
        when every stage-1 candidate failed.
    """
    logger.info(
        "  Stage 1: screening %d candidates at " "max_reweight_iterations=%d",
        len(SEARCH_GRID),
        screen_iters,
    )
    stage1 = _run_candidate_wave(
        eval_images,
        SEARCH_GRID,
        base_config,
        baseline.val_range,
        n_workers,
        screen_iters,
        threads,
        "screen",
    )
    finite = {sf: ent for sf, ent in stage1.items() if np.isfinite(ent)}
    if not finite:
        logger.error(
            "  !! every stage-1 candidate failed - using manual params"
        )
        return None

    finalists = sorted(finite, key=lambda sf: finite[sf])[:n_stage2]
    baseline_smoothness = MANUAL_PARAMS["smoothness_flatfield"]
    if baseline_smoothness not in finalists:
        finalists.append(baseline_smoothness)
    logger.info(
        "  Stage 2: re-scoring %d finalists at full fidelity",
        len(finalists),
    )
    stage2 = _run_candidate_wave(
        eval_images,
        finalists,
        base_config,
        baseline.val_range,
        n_workers,
        None,
        threads,
        "final",
    )
    # The baseline's full-fidelity score is already known exactly.
    stage2[baseline_smoothness] = baseline.entropy
    return stage2


def _finalize_report(
    report: dict[str, Any],
    stage2: dict[float, float],
    best_params: dict[str, float],
    best_entropy: float,
    baseline: Baseline,
) -> None:
    """
    Record the search outcome in ``report`` and log the verdict.

    Parameters
    ----------
    report : dict
        Search report, updated in place.
    stage2 : dict
        ``{smoothness_flatfield: entropy}`` at full fidelity.
    best_params : dict
        The selected parameters.
    best_entropy : float
        Entropy of the selected parameters.
    baseline : Baseline
        The incumbent, for the margin of the win.

    Returns
    -------
    None
    """
    report["ranked_stage2"] = sorted(
        ([float(sf), float(ent)] for sf, ent in stage2.items()),
        key=lambda pair: pair[1],
    )
    report["best_entropy"] = float(best_entropy)
    report["chose_baseline"] = best_params == dict(MANUAL_PARAMS)
    if report["chose_baseline"]:
        logger.info(
            "  No candidate beat the manual baseline -> keeping manual "
            "params."
        )
    else:
        logger.info(
            "  Winner: %s (beat baseline by %.2e nats)",
            best_params,
            baseline.entropy - best_entropy,
        )


def parallel_autotune(
    images: np.ndarray,
    base_config: dict[str, Any],
    tile_z_offsets: dict[int, tuple[int, int]] | None = None,
    max_eval_slices: int = 150,
    screen_iters: int = 3,
    n_stage2: int = 4,
    max_search_minutes: float = 45.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Parallel 1-D search for ``smoothness_flatfield``. MINIMIZES entropy.

    ``smoothness_darkfield`` and ``sparse_cost_darkfield`` are not
    searched: with ``get_darkfield=False`` they are provable no-ops, so
    the search is 1-D.

    Parameters
    ----------
    images : np.ndarray
        Pedestal-subtracted ``(N, H, W)`` fitting stack.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    tile_z_offsets : dict, optional
        Per-tile spans, so the evaluation subset is stratified by tile.
    max_eval_slices : int, optional
        Slices used to score each candidate, by default 150.
    screen_iters : int, optional
        Reduced iteration count for the stage-1 screen, by default 3.
    n_stage2 : int, optional
        Finalists re-scored at full fidelity, by default 4.
    max_search_minutes : float, optional
        Abort before searching if the projection exceeds this budget.

    Returns
    -------
    tuple of (dict, dict)
        The chosen parameters and a report of the search.
    """
    # Gets the worker pool size and the number of threads per worker.
    n_workers, threads = worker_pool_size()
    eval_images, n_eval = stratified_subsample(
        images, max_eval_slices, tile_z_offsets
    )
    logger.info(
        "  Autotune: %d/%d slices (stratified across %d tiles), %d "
        "workers x %d threads",
        n_eval,
        images.shape[0],
        len(tile_z_offsets or {}),
        n_workers,
        threads,
    )

    baseline = _fit_baseline(eval_images, base_config)
    baseline_smoothness = MANUAL_PARAMS["smoothness_flatfield"]
    report: dict[str, Any] = {
        "baseline_entropy": baseline.entropy,
        "baseline_std": baseline.std,
        "n_eval_slices": n_eval,
        "chose_baseline": True,
        "best_entropy": baseline.entropy,
        "ranked_stage2": [],
        "reason": None,
    }

    projected = _project_search_minutes(
        baseline, base_config, n_workers, screen_iters, n_stage2
    )
    if projected * 2 > max_search_minutes:
        logger.warning(
            "  !! projection exceeds the budget of %.0f min - using "
            "manual params (raise --max-search-minutes or lower "
            "--max-eval-slices to search anyway)",
            max_search_minutes,
        )
        report["reason"] = "over_budget"
        return dict(MANUAL_PARAMS), report

    stage2 = _screen_and_rescore(
        eval_images,
        base_config,
        baseline,
        n_workers,
        threads,
        screen_iters,
        n_stage2,
    )
    if stage2 is None:
        report["reason"] = "all_candidates_failed"
        return dict(MANUAL_PARAMS), report

    best_params, best_entropy = _select_winner(stage2, baseline.entropy)
    _log_ranking(stage2, baseline_smoothness, best_params)
    _finalize_report(report, stage2, best_params, best_entropy, baseline)
    return best_params, report


def confirm_against_baseline(
    images: np.ndarray,
    base_config: dict[str, Any],
    winner_params: dict[str, float],
    tile_z_offsets: dict[int, tuple[int, int]] | None = None,
    n_confirm: int = 1500,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Refit the winner head-to-head at closer-to-final scale.

    The search runs on ~150 slices but the final fit uses thousands, and
    in ladmap mode ``init_mu`` depends on N, so the optimal smoothness
    can genuinely shift with N.  Entropy is not comparable across
    different N, so this stage freezes its own window from its own
    baseline fit and scores both candidates inside it.  Reverts to the
    manual parameters unless the winner still leads.

    Parameters
    ----------
    images : np.ndarray
        Pedestal-subtracted fitting stack.
    base_config : dict
        BaSiC keyword arguments, used verbatim.
    winner_params : dict
        Parameters that won the search.
    tile_z_offsets : dict, optional
        Per-tile spans, so the confirmation subset is stratified.
    n_confirm : int, optional
        Slices for the confirmation refit, by default 1500.

    Returns
    -------
    tuple of (dict, dict)
        The parameters to use and a report of the head-to-head.
    """
    sub, n_sub = stratified_subsample(
        images, n_confirm, tile_z_offsets, seed=7
    )
    logger.info(
        "  Confirming at N=%d (the search used far fewer slices)", n_sub
    )

    incumbent = BaSiC(**{**base_config, **MANUAL_PARAMS})
    incumbent.fit(sub)
    transformed = np.asarray(incumbent.transform(sub), dtype=np.float64)
    vmin, vmax = np.quantile(transformed, [0.01, 0.99])
    val_range = float(vmax - vmin)
    entropy_base = basic_entropy(
        transformed, float(vmin), float(vmin) + val_range
    )
    std_base = float(np.asarray(incumbent.flatfield).std())
    del transformed

    challenger = BaSiC(**{**base_config, **winner_params})
    challenger.fit(sub)
    entropy_win = score_fit(challenger, sub, val_range)
    std_win = float(np.asarray(challenger.flatfield).std())

    logger.info(
        "    baseline entropy=%.6f (std=%.4f)   winner entropy=%.6f "
        "(std=%.4f)",
        entropy_base,
        std_base,
        entropy_win,
        std_win,
    )
    info = {
        "n": n_sub,
        "baseline_ent": entropy_base,
        "winner_ent": entropy_win,
        "baseline_std": std_base,
        "winner_std": std_win,
    }
    if entropy_win < entropy_base - SELECT_TOL:
        info["held_up"] = True
        return dict(winner_params), info
    logger.info(
        "    winner did not hold up at scale -> reverting to manual " "params."
    )
    info["held_up"] = False
    return dict(MANUAL_PARAMS), info
