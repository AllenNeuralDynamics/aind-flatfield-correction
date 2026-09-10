"""Tests for flatfield fitting and the plausibility guard."""

import unittest
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np

from aind_flatfield_correction.core.basicpy import fit
from aind_flatfield_correction.core.basicpy.config import BASIC_CONFIG
from tests.fakes import InlineExecutor, inline_as_completed, make_basic


def _plausible(height=8, width=8, span=0.5):
    """
    Build a flatfield that clears the guard.

    Parameters
    ----------
    height : int, optional
        Plane height, by default 8.
    width : int, optional
        Plane width, by default 8.
    span : float, optional
        Peak-to-peak width around 1.0, by default 0.5.

    Returns
    -------
    np.ndarray
        A ramp flatfield of the requested span.
    """
    ramp = np.linspace(1.0 - span / 2, 1.0 + span / 2, height * width)
    return ramp.reshape(height, width).astype(np.float32)


class TestFitOneZ(unittest.TestCase):
    """The per-Z worker."""

    def _run(self, stacks, zi, basic=None):
        """
        Run the worker against a real shared-memory block.

        Parameters
        ----------
        stacks : np.ndarray
            ``(n_planes, n_tiles, H, W)`` array to share.
        zi : int
            Z index to fit.
        basic : type, optional
            Fake BaSiC class, by default a passing one.

        Returns
        -------
        tuple
            The worker's ``(z_index, flatfield, traceback)``.
        """
        shm = shared_memory.SharedMemory(create=True, size=stacks.nbytes)
        try:
            view = np.ndarray(stacks.shape, dtype=stacks.dtype, buffer=shm.buf)
            view[:] = stacks
            with patch.object(fit, "BaSiC", basic or make_basic()):
                return fit._fit_one_z(
                    shm.name, stacks.shape, stacks.dtype, zi, BASIC_CONFIG
                )
        finally:
            shm.close()
            shm.unlink()

    def test_fits_one_z_index_across_all_tiles(self):
        """The slice handed to BaSiC is n_tiles images of one plane."""
        stacks = np.ones((3, 4, 8, 8), dtype=np.float32)
        zi, flat, err = self._run(stacks, 1)
        self.assertEqual(zi, 1)
        self.assertEqual(flat.shape, (8, 8))
        self.assertIsNone(err)

    def test_reports_a_failure_with_its_traceback(self):
        """The parent logs what went wrong in the worker."""
        stacks = np.ones((2, 2, 4, 4), dtype=np.float32)
        zi, flat, err = self._run(stacks, 0, basic=make_basic(fail=True))
        self.assertEqual(zi, 0)
        self.assertIsNone(flat)
        self.assertIn("fake fit failure", err)


class TestStackPlanesByZ(unittest.TestCase):
    """Regrouping the concatenated stack by Z index."""

    def test_gathers_plane_k_of_every_tile(self):
        """Plane k lives at start + k within each tile's span."""
        slices = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
        offsets = {0: (0, 4), 1: (4, 8), 2: (8, 12)}
        grouped = fit._stack_planes_by_z(slices, offsets, 4)
        self.assertEqual(grouped.shape, (4, 3, 1, 1))
        np.testing.assert_array_equal(
            grouped[:, :, 0, 0],
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
        )

    def test_repeats_the_last_plane_of_a_short_tile(self):
        """A tile with fewer planes must not read past its span."""
        slices = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
        offsets = {0: (0, 3), 1: (3, 5)}
        grouped = fit._stack_planes_by_z(slices, offsets, 3)
        np.testing.assert_array_equal(grouped[:, 1, 0, 0], [3, 4, 4])

    def test_orders_tiles_by_index(self):
        """Span keys are tile indices, so sorting them is the order."""
        slices = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
        grouped = fit._stack_planes_by_z(slices, {1: (2, 4), 0: (0, 2)}, 2)
        np.testing.assert_array_equal(grouped[0, :, 0, 0], [0, 2])


class TestCollectPerZFlatfields(unittest.TestCase):
    """Running the per-Z fits in parallel."""

    def _collect(self, n_planes, basic):
        """
        Collect per-Z fits with the pool running inline.

        Parameters
        ----------
        n_planes : int
            Number of Z fits to run.
        basic : type
            Fake BaSiC class to patch in.

        Returns
        -------
        dict
            ``{z_index: flatfield}`` for the fits that succeeded.
        """
        stacks = np.ones((n_planes, 3, 4, 4), dtype=np.float32)
        with patch.object(fit, "ProcessPoolExecutor", InlineExecutor):
            with patch.object(fit, "as_completed", inline_as_completed):
                with patch.object(fit, "BaSiC", basic):
                    return fit._collect_per_z_flatfields(
                        stacks, n_planes, BASIC_CONFIG
                    )

    def test_collects_one_flatfield_per_plane(self):
        """Six planes exercise the progress log as well."""
        flats = self._collect(6, make_basic())
        self.assertEqual(sorted(flats), list(range(6)))
        self.assertEqual(flats[0].shape, (4, 4))

    def test_drops_the_planes_whose_fit_failed(self):
        """The median is taken over whatever succeeded."""
        flats = self._collect(3, make_basic(fail=True))
        self.assertEqual(flats, {})


class TestEstimatePerZMedian(unittest.TestCase):
    """The per-Z-median estimator."""

    def test_combines_the_per_z_fits_by_median(self):
        """The median rejects planes whose fit went wrong."""
        slices = np.ones((6, 4, 4), dtype=np.float32)
        offsets = {0: (0, 3), 1: (3, 6)}
        flats = {0: _plausible(4, 4), 1: _plausible(4, 4) * 2}
        with patch.object(
            fit, "_collect_per_z_flatfields", return_value=flats
        ):
            flatfield, darkfield, n_ok = fit.estimate_per_z_median(
                slices, offsets, 3, BASIC_CONFIG
            )
        self.assertEqual(n_ok, 2)
        self.assertEqual(flatfield.shape, (4, 4))
        self.assertTrue(np.all(darkfield == 0))

    def test_raises_when_every_fit_failed(self):
        """There is no flatfield to fall back to here."""
        slices = np.ones((4, 4, 4), dtype=np.float32)
        with patch.object(fit, "_collect_per_z_flatfields", return_value={}):
            with self.assertRaises(RuntimeError):
                fit.estimate_per_z_median(
                    slices, {0: (0, 2), 1: (2, 4)}, 2, BASIC_CONFIG
                )


class TestEstimateJoint(unittest.TestCase):
    """The single joint fit."""

    def test_returns_the_fitted_fields(self):
        """No per-Z fits to count, so the third value is None."""
        slices = np.ones((5, 8, 8), dtype=np.float32)
        with patch.object(fit, "BaSiC", make_basic()):
            flatfield, darkfield, n_ok = fit.estimate_joint(
                slices, BASIC_CONFIG
            )
        self.assertEqual(flatfield.shape, (8, 8))
        self.assertEqual(darkfield.shape, (8, 8))
        self.assertIsNone(n_ok)

    def test_passes_the_configuration_through(self):
        """The chosen smoothness must reach the solver."""
        recorded = []
        with patch.object(fit, "BaSiC", make_basic(record=recorded)):
            fit.estimate_joint(
                np.ones((2, 4, 4), dtype=np.float32),
                {**BASIC_CONFIG, "smoothness_flatfield": 0.25},
            )
        self.assertEqual(recorded[0]["smoothness_flatfield"], 0.25)


class TestCheckFlatfieldPlausibility(unittest.TestCase):
    """The guard that catches a no-op or runaway flatfield."""

    def test_accepts_a_realistic_flatfield(self):
        """Known-good flats measured std 0.08-0.11, span 0.37-0.48."""
        ok, stats, reasons = fit.check_flatfield_plausibility(_plausible())
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(stats["mean"], 1.0, places=5)

    def test_rejects_a_flat_flatfield(self):
        """This is the check that catches an inverted objective."""
        ok, _, reasons = fit.check_flatfield_plausibility(np.ones((8, 8)))
        self.assertFalse(ok)
        self.assertEqual(len(reasons), 2)  # std floor and span floor
        self.assertTrue(any("no-op" in reason for reason in reasons))

    def test_rejects_non_finite_values(self):
        """A NaN would propagate through every corrected tile."""
        flat = _plausible()
        flat[0, 0] = np.nan
        ok, _, reasons = fit.check_flatfield_plausibility(flat)
        self.assertFalse(ok)
        self.assertIn("non-finite", reasons[0])

    def test_rejects_a_non_positive_minimum(self):
        """Dividing by zero would blow up the correction."""
        flat = _plausible()
        flat[0, 0] = 0.0
        ok, _, reasons = fit.check_flatfield_plausibility(flat)
        self.assertFalse(ok)
        self.assertTrue(any("blow up" in reason for reason in reasons))

    def test_rejects_a_runaway_flatfield(self):
        """An over-fit field is as wrong as a flat one."""
        ok, _, reasons = fit.check_flatfield_plausibility(_plausible(span=3.0))
        self.assertFalse(ok)
        self.assertTrue(any("runaway" in reason for reason in reasons))

    def test_rejects_a_flatfield_far_below_the_baseline(self):
        """A relative floor catches a field the absolute one allows."""
        flat = _plausible(span=0.12)
        ok, _, reasons = fit.check_flatfield_plausibility(
            flat, baseline_std=0.20
        )
        self.assertFalse(ok)
        self.assertTrue(any("baseline" in reason for reason in reasons))

    def test_ignores_the_relative_floor_without_a_baseline(self):
        """The absolute floors still apply on their own."""
        ok, _, _ = fit.check_flatfield_plausibility(
            _plausible(span=0.12), baseline_std=None
        )
        self.assertTrue(ok)


class TestReportFlatfield(unittest.TestCase):
    """The logging wrapper around the guard."""

    def test_passes_a_good_flatfield_through(self):
        """Same verdict as the guard, with the stats logged."""
        ok, stats, reasons = fit.report_flatfield(_plausible())
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        self.assertIn("span", stats)

    def test_flags_a_suspicious_flatfield(self):
        """The banner is what a reviewer notices in a long log."""
        ok, _, reasons = fit.report_flatfield(
            np.ones((4, 4)), label="(manual refit)"
        )
        self.assertFalse(ok)
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
