"""Tests for the flatfield correction metrics."""

import unittest
import warnings

import numpy as np

from aind_flatfield_correction.metrics.metrics import cv, masked_profile


class TestCv(unittest.TestCase):
    """The coefficient of variation used to score a correction."""

    def test_measures_relative_spread_as_a_percentage(self):
        """A flatter profile has to score lower."""
        flat = cv(np.array([100.0, 100.0, 100.0]))
        bowed = cv(np.array([50.0, 100.0, 150.0]))
        self.assertEqual(flat, 0.0)
        self.assertGreater(bowed, 0.0)

    def test_is_scale_free(self):
        """Doubling every value must not change the CV."""
        values = np.array([50.0, 100.0, 150.0])
        self.assertAlmostEqual(cv(values), cv(values * 2), places=10)

    def test_ignores_nans(self):
        """The profiles are masked, so empty positions arrive as NaN."""
        self.assertAlmostEqual(
            cv(np.array([50.0, 100.0, 150.0])),
            cv(np.array([50.0, np.nan, 100.0, 150.0, np.nan])),
            places=10,
        )

    def test_an_all_nan_profile_is_not_a_number(self):
        """A fully masked profile has no CV to report."""
        self.assertTrue(np.isnan(cv(np.array([np.nan, np.nan]))))

    def test_a_zero_mean_profile_is_not_a_number(self):
        """Guards the division rather than returning an infinity."""
        self.assertTrue(np.isnan(cv(np.array([-1.0, 1.0]))))


class TestMaskedProfile(unittest.TestCase):
    """Averaging over occupied pixels only."""

    def test_ignores_empty_positions(self):
        """Padding zeros would drag the mean towards zero."""
        image = np.array([[100.0, 0.0], [200.0, 0.0]])
        with warnings.catch_warnings():
            # The fully masked column is the point of the test; numpy
            # warns about the mean of an empty slice.
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = masked_profile(image, 0)
        np.testing.assert_array_equal(profile, [150.0, np.nan])

    def test_averages_along_the_requested_axis(self):
        """Axis 0 gives the X profile, axis 1 the Y profile."""
        image = np.array([[100.0, 200.0], [300.0, 400.0]])
        np.testing.assert_array_equal(masked_profile(image, 0), [200.0, 300.0])
        np.testing.assert_array_equal(masked_profile(image, 1), [150.0, 350.0])

    def test_an_all_empty_row_is_not_a_number(self):
        """A grid position with no tile contributes nothing."""
        image = np.zeros((2, 2))
        with warnings.catch_warnings():
            # numpy warns about the mean of an empty slice; a fully
            # masked profile returning NaN is the intended behaviour.
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = masked_profile(image, 0)
        self.assertTrue(np.all(np.isnan(profile)))
