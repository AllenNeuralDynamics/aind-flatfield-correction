"""Tests for the inspection figures."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aind_flatfield_correction.core.basicpy import figures
from aind_flatfield_correction.core.basicpy.tiles import FitStack

TILE_NAMES = [
    f"Tile_X_{i:04d}_Y_0000_Z_0000_ch_405.ome.zarr" for i in range(4)
]


def _vignetted_stack(n_tiles=4, planes=3, size=16):
    """
    Build a stack whose planes carry a radial intensity falloff.

    Parameters
    ----------
    n_tiles : int, optional
        Number of tiles, by default 4.
    planes : int, optional
        Planes per tile, by default 3.
    size : int, optional
        Plane height and width, by default 16.

    Returns
    -------
    tuple of (FitStack, np.ndarray, np.ndarray)
        The stack, the flatfield that flattens it, and the darkfield.
    """
    grid = np.linspace(-1.0, 1.0, size)
    x_grid, y_grid = np.meshgrid(grid, grid)
    falloff = 1.0 - 0.3 * (x_grid**2 + y_grid**2)
    flatfield = falloff.astype(np.float32)
    dark = np.full((size, size), 90.0, dtype=np.float32)

    slices = np.stack(
        [dark + 100.0 * falloff for _ in range(n_tiles * planes)]
    ).astype(np.float32)
    offsets = {
        index: (index * planes, (index + 1) * planes)
        for index in range(n_tiles)
    }
    stack = FitStack(slices, offsets, planes, size, size)
    return stack, flatfield, dark


class TestCorrectSlice(unittest.TestCase):
    """The visualization-only correction."""

    def test_removes_the_pedestal_then_divides(self):
        """Same order as the fit: subtract, clip, divide."""
        plane = np.array([[190.0]], dtype=np.float32)
        dark = np.array([[90.0]], dtype=np.float32)
        flat = np.array([[2.0]], dtype=np.float32)
        result = figures.correct_slice(plane, flat, dark)
        self.assertAlmostEqual(float(result[0, 0]), 50.0)

    def test_clips_below_the_pedestal_to_zero(self):
        """A pixel darker than the offset must not go negative."""
        plane = np.array([[10.0]], dtype=np.float32)
        dark = np.array([[90.0]], dtype=np.float32)
        flat = np.ones((1, 1), dtype=np.float32)
        self.assertEqual(figures.correct_slice(plane, flat, dark)[0, 0], 0.0)

    def test_guards_against_a_non_positive_flatfield(self):
        """A zero in the flatfield would divide by zero."""
        plane = np.array([[100.0]], dtype=np.float32)
        dark = np.zeros((1, 1), dtype=np.float32)
        flat = np.zeros((1, 1), dtype=np.float32)
        result = figures.correct_slice(plane, flat, dark)
        self.assertTrue(np.isfinite(result).all())


class TestPickSampleTiles(unittest.TestCase):
    """Choosing which tiles get a profile figure."""

    def test_spreads_the_picks_across_the_list(self):
        """Acquisition order tends to sweep the mosaic."""
        self.assertEqual(figures.pick_sample_tiles(10, 3), [0, 4, 9])

    def test_returns_every_tile_when_asked_for_more(self):
        """Never index past the end of the list."""
        self.assertEqual(figures.pick_sample_tiles(3, 9), [0, 1, 2])

    def test_handles_an_empty_tile_list(self):
        """A guard against dividing an empty spread."""
        self.assertEqual(figures.pick_sample_tiles(0, 4), [])

    def test_handles_a_request_for_no_tiles(self):
        """--validate-tiles 0 asks for no per-tile figures."""
        self.assertEqual(figures.pick_sample_tiles(10, 0), [])

    def test_never_repeats_an_index(self):
        """Rounding can land two picks on the same tile."""
        picks = figures.pick_sample_tiles(4, 3)
        self.assertEqual(len(picks), len(set(picks)))


class TestSafeName(unittest.TestCase):
    """Turning a tile name into a filename fragment."""

    def test_strips_the_ome_zarr_suffix(self):
        """The suffix is noise in a figure filename."""
        self.assertEqual(
            figures._safe_name("Tile_X_0001_ch_405.ome.zarr"),
            "Tile_X_0001_ch_405",
        )

    def test_strips_a_trailing_slash(self):
        """S3 listings can carry one."""
        self.assertEqual(figures._safe_name("tile_a.ome.zarr/"), "tile_a")

    def test_collapses_unsafe_characters(self):
        """Spaces and punctuation must not reach the filesystem."""
        self.assertEqual(
            figures._safe_name("tile a/b (2).ome.zarr"), "tile_a_b_2"
        )


class TestDisplayHelpers(unittest.TestCase):
    """Display scaling and plane selection."""

    def test_display_range_ignores_empty_pixels(self):
        """Padding zeros would drag the low end down."""
        image = np.concatenate(
            [np.zeros(50), np.linspace(100, 200, 50)]
        ).reshape(10, 10)
        low, high = figures._display_range(image)
        self.assertGreater(low, 90.0)
        self.assertLessEqual(high, 200.0)

    def test_display_range_of_an_empty_image(self):
        """An all-zero plane still has to render."""
        self.assertEqual(figures._display_range(np.zeros((4, 4))), (0.0, 1.0))

    def test_display_plane_takes_a_fractional_depth(self):
        """Z_FRACTION through the tile's own span, not the stack."""
        slices = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
        stack = FitStack(slices, {0: (0, 4), 1: (4, 8)}, 4, 1, 1)
        self.assertEqual(figures._display_plane(stack, 0)[0, 0], 1.0)
        self.assertEqual(figures._display_plane(stack, 1)[0, 0], 5.0)


class TestPlotVignetting(unittest.TestCase):
    """The mean-plane vignetting figure."""

    def test_writes_the_figure_and_measures_the_improvement(self):
        """A correct flatfield must lower the profile's CV."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "vignetting.png"
            metrics = figures._plot_vignetting(
                "ch_405",
                stack.slices.mean(axis=0),
                flatfield,
                dark,
                out,
            )
            self.assertTrue(out.exists())
        self.assertLess(metrics["cv_x_after"], metrics["cv_x_before"])
        self.assertLess(metrics["cv_y_after"], metrics["cv_y_before"])

    def test_reports_both_axes(self):
        """X and Y vignetting are independent."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            metrics = figures._plot_vignetting(
                "ch_405",
                stack.slices.mean(axis=0),
                flatfield,
                dark,
                Path(tmp) / "v.png",
            )
        self.assertEqual(
            sorted(metrics),
            ["cv_x_after", "cv_x_before", "cv_y_after", "cv_y_before"],
        )


class TestPlotFigures(unittest.TestCase):
    """The overview and per-tile panels."""

    def test_overview_is_written(self):
        """Flatfield beside one example plane, raw and corrected."""
        stack, flatfield, dark = _vignetted_stack()
        plane = figures._display_plane(stack, 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "overview.png"
            figures._plot_overview(
                "ch_405",
                flatfield,
                plane,
                figures.correct_slice(plane, flatfield, dark),
                3,
                out,
            )
            self.assertGreater(out.stat().st_size, 0)

    def test_tile_profile_is_written(self):
        """The per-tile X/Y profiles are the acceptance gate."""
        stack, flatfield, dark = _vignetted_stack()
        plane = figures._display_plane(stack, 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tile.png"
            figures._plot_tile_profile(
                "ch_405",
                TILE_NAMES[1],
                plane,
                figures.correct_slice(plane, flatfield, dark),
                dark,
                out,
            )
            self.assertGreater(out.stat().st_size, 0)


class TestSaveValidationFigures(unittest.TestCase):
    """The whole figure set for one run."""

    def test_writes_a_profile_figure_per_sampled_tile(self):
        """One slice each from n different tiles."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ch_405"
            metrics = figures.save_validation_figures(
                "ch_405",
                flatfield,
                dark,
                stack,
                TILE_NAMES,
                out_dir,
                3,
                2,
            )
            written = sorted(path.name for path in out_dir.iterdir())
        self.assertEqual(len(written), 4)  # 2 tiles + overview + vignetting
        self.assertIn("ch_405_overview.png", written)
        self.assertIn("ch_405_vignetting_profiles.png", written)
        self.assertEqual(len([name for name in written if "_tile" in name]), 2)
        self.assertEqual(len(metrics["figures"]), 4)

    def test_names_the_figures_after_their_tile(self):
        """A reviewer has to know which tile they are looking at."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ch_405"
            figures.save_validation_figures(
                "ch_405",
                flatfield,
                dark,
                stack,
                TILE_NAMES,
                out_dir,
                3,
                1,
            )
            names = [path.name for path in out_dir.iterdir()]
        self.assertTrue(any("Tile_X_0000" in name for name in names), names)

    def test_creates_the_output_folder(self):
        """The per-run figure folder does not exist yet."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "ch_405"
            figures.save_validation_figures(
                "ch_405",
                flatfield,
                dark,
                stack,
                TILE_NAMES,
                out_dir,
                3,
                1,
            )
            self.assertTrue(out_dir.is_dir())

    def test_returns_the_correction_metrics(self):
        """The CV numbers end up in the sidecar."""
        stack, flatfield, dark = _vignetted_stack()
        with tempfile.TemporaryDirectory() as tmp:
            metrics = figures.save_validation_figures(
                "ch_405",
                flatfield,
                dark,
                stack,
                TILE_NAMES,
                Path(tmp) / "ch",
                3,
                1,
            )
        self.assertLess(metrics["cv_x_after"], metrics["cv_x_before"])


if __name__ == "__main__":
    unittest.main()
