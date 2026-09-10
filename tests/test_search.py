"""Tests for the entropy objective and the parameter search."""

import unittest
from unittest.mock import patch

import numpy as np

from aind_flatfield_correction.core.basicpy import search
from aind_flatfield_correction.core.basicpy.config import (
    BASIC_CONFIG,
    baseline_params,
    search_grid,
)
from tests.fakes import InlineExecutor, inline_as_completed, make_basic

INCUMBENT = baseline_params(BASIC_CONFIG)
BASELINE_SF = INCUMBENT["smoothness_flatfield"]
GRID = search_grid(BASELINE_SF)


def _images(n_slices=8, size=6, seed=0):
    """
    Build a small deterministic image stack.

    Parameters
    ----------
    n_slices : int, optional
        Number of planes, by default 8.
    size : int, optional
        Plane height and width, by default 6.
    seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    np.ndarray
        A ``(n_slices, size, size)`` float32 stack.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(100, 10, (n_slices, size, size)).astype(np.float32)


def _offsets(n_tiles=4, per_tile=2):
    """
    Build index-keyed per-tile spans.

    Parameters
    ----------
    n_tiles : int, optional
        Number of tiles, by default 4.
    per_tile : int, optional
        Planes per tile, by default 2.

    Returns
    -------
    dict
        ``{index: (start, end)}`` spans.
    """
    return {
        index: (index * per_tile, (index + 1) * per_tile)
        for index in range(n_tiles)
    }


class TestBasicEntropy(unittest.TestCase):
    """The objective replicated from basicpy, minimized not maximized."""

    def test_matches_a_known_distribution(self):
        """A standard normal over [-3, 3] is about 1.4 nats."""
        values = np.random.default_rng(0).normal(size=40000)
        entropy = search.basic_entropy(values, -3, 3)
        self.assertAlmostEqual(entropy, 1.41, places=1)

    def test_a_narrower_spread_scores_lower(self):
        """Lower is better, so a corrected image must win."""
        rng = np.random.default_rng(1)
        narrow = search.basic_entropy(rng.normal(scale=0.2, size=20000), -3, 3)
        wide = search.basic_entropy(rng.normal(scale=1.5, size=20000), -3, 3)
        self.assertLess(narrow, wide)

    def test_drops_out_of_range_values_instead_of_clamping(self):
        """basicpy uses clip=True, which discards rather than clamps."""
        inside = np.array([0.0, 0.1, 0.2, 0.3])
        with_outliers = np.concatenate([inside, [500.0, -500.0]])
        self.assertEqual(
            search.basic_entropy(inside, -1, 1),
            search.basic_entropy(with_outliers, -1, 1),
        )

    def test_ignores_non_finite_values(self):
        """A NaN in the transform must not poison the score."""
        clean = np.array([0.0, 0.5, 1.0])
        dirty = np.array([0.0, 0.5, 1.0, np.nan, np.inf])
        self.assertEqual(
            search.basic_entropy(clean, -2, 2),
            search.basic_entropy(dirty, -2, 2),
        )

    def test_returns_the_losing_sentinel_when_empty(self):
        """+inf is the losing value under argmin."""
        values = np.array([100.0, 200.0])
        self.assertEqual(search.basic_entropy(values, -1, 1), float("inf"))


class TestScoreFit(unittest.TestCase):
    """Scoring a fitted model inside the frozen window."""

    def test_scores_the_transform_of_the_images(self):
        """The window width is frozen; only its lower edge moves."""
        images = _images()
        basic = make_basic()(smoothness_flatfield=0.5)
        basic.fit(images)
        score = search.score_fit(basic, images, 40.0)
        self.assertTrue(np.isfinite(score))

    def test_a_tighter_candidate_scores_lower(self):
        """The fake spreads by smoothness, so small smoothness wins."""
        images = _images()
        tight = make_basic()(smoothness_flatfield=0.01)
        loose = make_basic()(smoothness_flatfield=10.0)
        tight.fit(images)
        loose.fit(images)
        self.assertLess(
            search.score_fit(tight, images, 40.0),
            search.score_fit(loose, images, 40.0),
        )


class TestStratifiedSubsample(unittest.TestCase):
    """Drawing evaluation slices equally from every tile."""

    def test_returns_everything_when_under_target(self):
        """No sampling needed, and no copy made."""
        images = _images(4)
        sub, count = search.stratified_subsample(images, 10)
        self.assertIs(sub, images)
        self.assertEqual(count, 4)

    def test_draws_from_every_tile(self):
        """A uniform draw would over-weight whichever tiles it hit."""
        images = _images(8)
        offsets = _offsets(4, 2)
        sub, count = search.stratified_subsample(images, 4, offsets)
        self.assertEqual(count, 4)
        self.assertEqual(sub.shape[1:], images.shape[1:])

    def test_trims_evenly_when_the_draw_overshoots(self):
        """Ceil-per-tile can exceed the target for many tiles."""
        images = _images(30)
        offsets = _offsets(10, 3)
        _, count = search.stratified_subsample(images, 7, offsets)
        self.assertEqual(count, 7)

    def test_skips_a_tile_with_an_empty_span(self):
        """A tile that contributed no planes must not break the draw."""
        images = _images(8)
        offsets = {0: (0, 4), 1: (4, 4), 2: (4, 8)}
        _, count = search.stratified_subsample(images, 4, offsets)
        self.assertGreater(count, 0)

    def test_falls_back_to_a_uniform_draw(self):
        """Without spans there is nothing to stratify by."""
        images = _images(20)
        sub, count = search.stratified_subsample(images, 5)
        self.assertEqual(count, 5)
        self.assertEqual(sub.shape[0], 5)

    def test_is_reproducible(self):
        """The same seed must give the same evaluation set."""
        images = _images(20)
        first, _ = search.stratified_subsample(images, 5, _offsets(4, 5))
        second, _ = search.stratified_subsample(images, 5, _offsets(4, 5))
        np.testing.assert_array_equal(first, second)


class TestEvalBasicParams(unittest.TestCase):
    """The worker that fits and scores one candidate."""

    def _run(self, images, params, screen_iters=None, basic=None):
        """
        Run the worker against a real shared-memory block.

        Parameters
        ----------
        images : np.ndarray
            Images to share with the worker.
        params : dict
            Candidate parameters.
        screen_iters : int, optional
            Reduced iteration count, by default None.
        basic : type, optional
            Fake BaSiC class to patch in, by default a passing one.

        Returns
        -------
        tuple
            The worker's ``(entropy, params, traceback)``.
        """
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(create=True, size=images.nbytes)
        try:
            view = np.ndarray(images.shape, dtype=images.dtype, buffer=shm.buf)
            view[:] = images
            with patch.object(search, "BaSiC", basic or make_basic()):
                return search._eval_basic_params(
                    shm.name,
                    images.shape,
                    images.dtype,
                    params,
                    BASIC_CONFIG,
                    40.0,
                    screen_iters,
                )
        finally:
            shm.close()
            shm.unlink()

    def test_scores_a_candidate(self):
        """A successful fit returns a finite entropy and no traceback."""
        entropy, params, err = self._run(
            _images(), {"smoothness_flatfield": 0.5}
        )
        self.assertTrue(np.isfinite(entropy))
        self.assertEqual(params, {"smoothness_flatfield": 0.5})
        self.assertIsNone(err)

    def test_reports_a_failure_as_the_losing_sentinel(self):
        """A swallowed exception once masqueraded as a real search."""
        entropy, _, err = self._run(
            _images(),
            {"smoothness_flatfield": 0.5},
            basic=make_basic(fail=True),
        )
        self.assertEqual(entropy, float("inf"))
        self.assertIn("fake fit failure", err)

    def test_the_screen_lowers_the_iteration_count(self):
        """Stage 1 is cheap; it ranks but never decides."""
        recorded = []
        self._run(
            _images(),
            {"smoothness_flatfield": 0.5},
            screen_iters=3,
            basic=make_basic(record=recorded),
        )
        self.assertEqual(recorded[0]["max_reweight_iterations"], 3)

    def test_full_fidelity_keeps_the_configured_iterations(self):
        """Stage 2 must score under the solver that will be applied."""
        recorded = []
        self._run(
            _images(),
            {"smoothness_flatfield": 0.5},
            basic=make_basic(record=recorded),
        )
        self.assertEqual(
            recorded[0]["max_reweight_iterations"],
            BASIC_CONFIG["max_reweight_iterations"],
        )


class TestRunCandidateWave(unittest.TestCase):
    """Scoring a batch of candidates."""

    def test_scores_every_candidate(self):
        """One entropy per candidate, keyed by smoothness."""
        with patch.object(search, "ProcessPoolExecutor", InlineExecutor):
            with patch.object(search, "as_completed", inline_as_completed):
                with patch.object(search, "BaSiC", make_basic()):
                    scores = search._run_candidate_wave(
                        _images(),
                        [0.1, 1.0, 10.0],
                        BASIC_CONFIG,
                        40.0,
                        2,
                        None,
                        1,
                        "test",
                    )
        self.assertEqual(sorted(scores), [0.1, 1.0, 10.0])
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))

    def test_a_failing_candidate_gets_the_losing_sentinel(self):
        """The wave must survive one candidate blowing up."""
        with patch.object(search, "ProcessPoolExecutor", InlineExecutor):
            with patch.object(search, "as_completed", inline_as_completed):
                with patch.object(search, "BaSiC", make_basic(fail=True)):
                    scores = search._run_candidate_wave(
                        _images(), [1.0], BASIC_CONFIG, 40.0, 1, 3, 1, "t"
                    )
        self.assertEqual(scores[1.0], float("inf"))


class TestFitBaseline(unittest.TestCase):
    """The reference fit that anchors the search."""

    def test_freezes_the_window_and_records_the_cost(self):
        """One fit yields the window, the incumbent score and a timing."""
        with patch.object(search, "BaSiC", make_basic()):
            baseline = search._fit_baseline(_images(), BASIC_CONFIG)
        self.assertGreater(baseline.val_range, 0.0)
        self.assertTrue(np.isfinite(baseline.entropy))
        self.assertGreater(baseline.std, 0.0)
        self.assertGreaterEqual(baseline.fit_seconds, 0.0)

    def test_fits_with_the_configured_params(self):
        """The incumbent is whatever the configuration carries."""
        recorded = []
        with patch.object(search, "BaSiC", make_basic(record=recorded)):
            search._fit_baseline(_images(), BASIC_CONFIG)
        self.assertEqual(recorded[0]["smoothness_flatfield"], BASELINE_SF)

    def test_fits_a_supplied_smoothness(self):
        """A run can hand its own hand-tuned value to the search."""
        recorded = []
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        with patch.object(search, "BaSiC", make_basic(record=recorded)):
            search._fit_baseline(_images(), config)
        self.assertEqual(recorded[0]["smoothness_flatfield"], 2.5)


class TestProjectSearchMinutes(unittest.TestCase):
    """The calibration gate's cost projection."""

    def test_scales_with_the_measured_fit_cost(self):
        """Twice the per-fit cost is twice the projection."""
        cheap = search.Baseline(1.0, 0.5, 0.1, 60.0)
        pricey = search.Baseline(1.0, 0.5, 0.1, 120.0)
        args = (BASIC_CONFIG, 8, 3, 4, len(GRID))
        self.assertAlmostEqual(
            2 * search._project_search_minutes(cheap, *args),
            search._project_search_minutes(pricey, *args),
            places=6,
        )

    def test_more_workers_project_a_shorter_search(self):
        """Waves shrink as the pool grows."""
        baseline = search.Baseline(1.0, 0.5, 0.1, 60.0)
        few = search._project_search_minutes(
            baseline, BASIC_CONFIG, 1, 3, 4, len(GRID)
        )
        many = search._project_search_minutes(
            baseline, BASIC_CONFIG, 8, 3, 4, len(GRID)
        )
        self.assertGreater(few, many)


class TestSelectWinner(unittest.TestCase):
    """Argmin with the baseline as the incumbent."""

    def test_keeps_the_baseline_without_a_real_win(self):
        """A win inside the tolerance is noise, not an improvement."""
        scores = {BASELINE_SF: 1.0, 0.5: 1.0 - search.SELECT_TOL / 2}
        params, entropy = search._select_winner(scores, 1.0, INCUMBENT)
        self.assertEqual(params, dict(INCUMBENT))
        self.assertEqual(entropy, 1.0)

    def test_takes_a_candidate_that_beats_the_tolerance(self):
        """Lower entropy is a better correction."""
        scores = {BASELINE_SF: 1.0, 0.5: 0.5}
        params, entropy = search._select_winner(scores, 1.0, INCUMBENT)
        self.assertEqual(params, {"smoothness_flatfield": 0.5})
        self.assertEqual(entropy, 0.5)

    def test_never_selects_a_failed_candidate(self):
        """+inf must lose."""
        scores = {BASELINE_SF: 1.0, 0.5: float("inf")}
        params, _ = search._select_winner(scores, 1.0, INCUMBENT)
        self.assertEqual(params, dict(INCUMBENT))

    def test_keeps_a_supplied_incumbent(self):
        """The fallback is the configured value, not the built-in one."""
        supplied = {"smoothness_flatfield": 2.5}
        params, _ = search._select_winner({2.5: 1.0}, 1.0, supplied)
        self.assertEqual(params, supplied)


class TestScreenAndRescore(unittest.TestCase):
    """The two-stage screen."""

    def _stage2(self, scores_by_wave):
        """
        Run the screen with canned wave results.

        Parameters
        ----------
        scores_by_wave : list of dict
            One score dict per wave call.

        Returns
        -------
        dict or None
            The stage-2 scores.
        """
        baseline = search.Baseline(40.0, 0.9, 0.1, 1.0)
        with patch.object(
            search, "_run_candidate_wave", side_effect=scores_by_wave
        ):
            return search._screen_and_rescore(
                _images(), BASIC_CONFIG, baseline, 8, 1, 3, 2
            )

    def test_returns_none_when_every_candidate_fails(self):
        """An all-failures run must not look like a real search."""
        failed = {sf: float("inf") for sf in GRID}
        self.assertIsNone(self._stage2([failed]))

    def test_rescores_the_finalists_plus_the_baseline(self):
        """The incumbent is always re-scored at full fidelity."""
        stage1 = {sf: 1.0 + sf for sf in GRID}
        stage2 = {0.01: 0.8, 0.021544: 0.9}
        result = self._stage2([stage1, dict(stage2)])
        self.assertIn(BASELINE_SF, result)
        self.assertEqual(result[BASELINE_SF], 0.9)

    def test_screens_a_grid_that_includes_a_supplied_incumbent(self):
        """An off-grid hand-tuned value becomes an extra candidate."""
        baseline = search.Baseline(40.0, 0.9, 0.1, 1.0)
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        with patch.object(
            search,
            "_run_candidate_wave",
            side_effect=[{2.5: 0.8}, {2.5: 0.8}],
        ) as wave:
            search._screen_and_rescore(_images(), config, baseline, 8, 1, 3, 2)
        self.assertIn(2.5, wave.call_args_list[0][0][1])

    def test_uses_the_exact_baseline_score(self):
        """The baseline's full-fidelity entropy is already known."""
        stage1 = {sf: 1.0 for sf in GRID}
        result = self._stage2([stage1, {BASELINE_SF: 99.0}])
        self.assertEqual(result[BASELINE_SF], 0.9)


class TestFinalizeReport(unittest.TestCase):
    """Recording the outcome of the search."""

    def test_ranks_candidates_by_entropy(self):
        """The report is read by a human, so order it best-first."""
        report = {}
        baseline = search.Baseline(40.0, 1.0, 0.1, 1.0)
        search._finalize_report(
            report,
            {1.0: 1.0, 0.5: 0.4},
            {"smoothness_flatfield": 0.5},
            0.4,
            baseline,
            INCUMBENT,
        )
        self.assertEqual(report["ranked_stage2"][0], [0.5, 0.4])
        self.assertFalse(report["chose_baseline"])

    def test_marks_a_baseline_win(self):
        """Nothing beat the hand-tuned parameters."""
        report = {}
        baseline = search.Baseline(40.0, 1.0, 0.1, 1.0)
        search._finalize_report(
            report, {1.0: 1.0}, dict(INCUMBENT), 1.0, baseline, INCUMBENT
        )
        self.assertTrue(report["chose_baseline"])
        self.assertEqual(report["best_entropy"], 1.0)


class TestParallelAutotune(unittest.TestCase):
    """The whole search, end to end."""

    def test_abandons_a_search_over_budget(self):
        """A slow fit means the search would never finish."""
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search,
                "_fit_baseline",
                return_value=search.Baseline(40.0, 1.0, 0.1, 600.0),
            ):
                params, report = search.parallel_autotune(
                    _images(),
                    BASIC_CONFIG,
                    tile_z_offsets=_offsets(),
                    max_search_minutes=1.0,
                )
        self.assertEqual(params, dict(INCUMBENT))
        self.assertEqual(report["reason"], "over_budget")

    def test_falls_back_when_every_candidate_fails(self):
        """The known-good parameters are the safety net."""
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search, "_screen_and_rescore", return_value=None
            ):
                params, report = search.parallel_autotune(
                    _images(), BASIC_CONFIG, tile_z_offsets=_offsets()
                )
        self.assertEqual(params, dict(INCUMBENT))
        self.assertEqual(report["reason"], "all_candidates_failed")

    def test_selects_a_winner_that_beats_the_baseline(self):
        """A genuinely lower entropy displaces the incumbent."""
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search,
                "_screen_and_rescore",
                return_value={BASELINE_SF: 1.0, 0.01: 0.2},
            ):
                params, report = search.parallel_autotune(
                    _images(), BASIC_CONFIG, tile_z_offsets=_offsets()
                )
        self.assertEqual(params, {"smoothness_flatfield": 0.01})
        self.assertFalse(report["chose_baseline"])
        self.assertEqual(report["best_entropy"], 0.2)

    def test_falls_back_to_the_configured_smoothness(self):
        """Not to the built-in default the run chose to override."""
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search, "_screen_and_rescore", return_value=None
            ):
                params, _ = search.parallel_autotune(
                    _images(), config, tile_z_offsets=_offsets()
                )
        self.assertEqual(params, {"smoothness_flatfield": 2.5})

    def test_records_the_incumbent_in_the_report(self):
        """The sidecar says what the candidates were measured against."""
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search, "_screen_and_rescore", return_value=None
            ):
                _, report = search.parallel_autotune(
                    _images(), config, tile_z_offsets=_offsets()
                )
        self.assertEqual(
            report["baseline_params"], {"smoothness_flatfield": 2.5}
        )

    def test_reports_the_evaluation_size(self):
        """The report records how much data the choice rests on."""
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search, "_screen_and_rescore", return_value=None
            ):
                _, report = search.parallel_autotune(
                    _images(20),
                    BASIC_CONFIG,
                    tile_z_offsets=_offsets(4, 5),
                    max_eval_slices=8,
                )
        self.assertEqual(report["n_eval_slices"], 8)


class TestConfirmAgainstBaseline(unittest.TestCase):
    """The head-to-head refit at closer-to-final scale."""

    def _confirm(self, baseline_entropy, winner_entropy):
        """
        Run the confirmation with both scores pinned.

        Entropy is not comparable across different N, so this stage
        freezes its own window; the scores themselves are pinned here so
        the test covers the decision, not the objective.

        Parameters
        ----------
        baseline_entropy : float
            Score the incumbent gets.
        winner_entropy : float
            Score the challenger gets.

        Returns
        -------
        tuple of (dict, dict)
            The chosen parameters and the head-to-head report.
        """
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(
                search, "basic_entropy", return_value=baseline_entropy
            ):
                with patch.object(
                    search, "score_fit", return_value=winner_entropy
                ):
                    return search.confirm_against_baseline(
                        _images(20),
                        BASIC_CONFIG,
                        {"smoothness_flatfield": 0.01},
                        tile_z_offsets=_offsets(4, 5),
                        n_confirm=8,
                    )

    def test_keeps_a_winner_that_holds_up(self):
        """init_mu depends on N, so the win must be re-checked."""
        params, info = self._confirm(1.0, 0.5)
        self.assertEqual(params, {"smoothness_flatfield": 0.01})
        self.assertTrue(info["held_up"])
        self.assertEqual(info["n"], 8)

    def test_reverts_a_winner_that_does_not_hold_up(self):
        """No margin at scale means the small-subset win was noise."""
        params, info = self._confirm(1.0, 1.0)
        self.assertEqual(params, dict(INCUMBENT))
        self.assertFalse(info["held_up"])

    def test_reverts_a_win_inside_the_tolerance(self):
        """The margin must exceed SELECT_TOL, as in the main search."""
        params, _ = self._confirm(1.0, 1.0 - search.SELECT_TOL / 2)
        self.assertEqual(params, dict(INCUMBENT))

    def test_reverts_to_a_supplied_smoothness(self):
        """The incumbent to fall back to comes from the config."""
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        with patch.object(search, "BaSiC", make_basic()):
            with patch.object(search, "basic_entropy", return_value=1.0):
                with patch.object(search, "score_fit", return_value=1.0):
                    params, _ = search.confirm_against_baseline(
                        _images(20),
                        config,
                        {"smoothness_flatfield": 0.01},
                        tile_z_offsets=_offsets(4, 5),
                        n_confirm=8,
                    )
        self.assertEqual(params, {"smoothness_flatfield": 2.5})

    def test_records_both_scores(self):
        """The sidecar keeps the head-to-head for later review."""
        _, info = self._confirm(1.0, 0.5)
        self.assertEqual(info["baseline_ent"], 1.0)
        self.assertEqual(info["winner_ent"], 0.5)


if __name__ == "__main__":
    unittest.main()
