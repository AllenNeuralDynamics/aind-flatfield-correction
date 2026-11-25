"""Tests for the darkfield built from the darkest tiles."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aind_flatfield_correction.core.basicpy import darkmap
from tests.fakes import LazyArray

TILES = [f"Tile_X_{i:04d}_Y_0000_Z_0000_ch_488.ome.zarr" for i in range(5)]
# Darkest coarse plane of each tile: (plane index, level).
DARKEST = [(2, 92.0), (0, 90.0), (5, 95.0), (3, 91.0), (1, 100.0)]
COARSE_DEPTH, FACTOR, SIZE = 6, 8, 8


def _pattern(level):
    """
    Full-resolution dark plane of a tile whose darkest plane is ``level``.

    The column pattern has a mean of exactly zero (multiples of 0.125 are
    exact in float32), so the plane's mean is exactly ``level``, as the
    coarse plane is the mean of the planes under it.

    Parameters
    ----------
    level : float
        Mean of the plane.

    Returns
    -------
    np.ndarray
        An ``(8, 8)`` plane.
    """
    columns = np.tile(np.arange(SIZE, dtype=np.float32), (SIZE, 1))
    return (level + 0.25 * (columns - 3.5)).astype(np.float32)


def _tile_arrays(darkest=DARKEST):
    """
    Build coarse and full-resolution stacks for every tile.

    Coarse planes are 200 counts except the darkest one.  At full
    resolution, the planes under the darkest coarse plane hold the tile's
    dark plane (:func:`_pattern`) and every other plane is 500 counts, so
    reading the wrong planes would show.

    Parameters
    ----------
    darkest : list of tuple of (int, float), optional
        Darkest plane index and level of each tile.

    Returns
    -------
    dict
        ``(tile name, level)`` to the array :func:`darkmap.open_tile`
        returns.
    """
    arrays = {}
    for name, (plane, level) in zip(TILES, darkest):
        coarse = np.full((COARSE_DEPTH, 4, 4), 200.0, dtype=np.float32)
        coarse[plane] = level
        full = np.full((COARSE_DEPTH * FACTOR, SIZE, SIZE), 500.0, np.float32)
        full[slice(plane * FACTOR, (plane + 1) * FACTOR)] = _pattern(level)
        arrays[(name, "3")] = coarse
        arrays[(name, "0")] = full
    return arrays


def _opener(arrays):
    """
    Serve the test arrays the way ``open_tile`` would.

    Parameters
    ----------
    arrays : dict
        Output of :func:`_tile_arrays`.

    Returns
    -------
    callable
        A stand-in for ``open_tile``.
    """

    def open_tile(base_path, name, level):
        """
        Return one tile at one level.

        Parameters
        ----------
        base_path : str
            Ignored.
        name : str
            Tile name.
        level : int or str
            Pyramid level.

        Returns
        -------
        LazyArray
            The tile's stack at that level.
        """
        return LazyArray(arrays[(name, str(level))])

    return open_tile


class TestPlaneLevels(unittest.TestCase):
    """The mean of every plane of every tile."""

    def test_measures_each_plane(self):
        """One mean per plane, in Z order."""
        with patch.object(darkmap, "open_tile", _opener(_tile_arrays())):
            levels = darkmap.plane_levels("/data", TILES[:2], 3, subsample=2)
        self.assertEqual(len(levels), 2)
        np.testing.assert_allclose(levels[0], [200, 200, 92, 200, 200, 200])
        np.testing.assert_allclose(levels[1], [90, 200, 200, 200, 200, 200])


class TestDarkestTiles(unittest.TestCase):
    """Choosing the tiles whose darkest plane is darkest."""

    def test_darkest_first(self):
        """Tile, plane and level, sorted by the level."""
        levels = [np.array([5.0, 3.0]), np.array([1.0, 4.0]), np.array([9.0])]
        self.assertEqual(
            darkmap.darkest_tiles(levels, 2), [(1, 0, 1.0), (0, 1, 3.0)]
        )

    def test_all_tiles_when_there_are_fewer(self):
        """Asking for more tiles than exist uses them all."""
        levels = [np.array([5.0]), np.array([1.0])]
        self.assertEqual(len(darkmap.darkest_tiles(levels, 12)), 2)


class TestLightSpread(unittest.TestCase):
    """How much the darkest-plane level changes across tiles."""

    def test_percentiles_and_spread(self):
        """The spread is the 95th minus the 5th percentile."""
        spread = darkmap.light_spread(np.arange(101, dtype=float))
        self.assertEqual(spread["p5"], 5.0)
        self.assertEqual(spread["median"], 50.0)
        self.assertEqual(spread["p95"], 95.0)
        self.assertEqual(spread["spread"], 90.0)


class TestFullResolutionMedian(unittest.TestCase):
    """The full-resolution planes under one coarse plane."""

    def test_reads_the_planes_under_the_coarse_plane(self):
        """Coarse plane 3 of a depth-6 tile is full-resolution 24-31."""
        arrays = _tile_arrays()
        with patch.object(darkmap, "open_tile", _opener(arrays)):
            median, info = darkmap.full_resolution_median(
                "/data", TILES[3], 3, COARSE_DEPTH
            )
        np.testing.assert_array_equal(median, _pattern(91.0))
        self.assertEqual(info["full_resolution_planes"], [24, 32])
        self.assertEqual(info["tile"], TILES[3])
        self.assertLess(info["temporal_variance"], 0.001)

    def test_a_pyramid_that_keeps_the_depth(self):
        """One full-resolution plane per coarse plane."""
        full = np.arange(6, dtype=np.float32)[:, None, None] * np.ones(
            (6, 2, 2), dtype=np.float32
        )
        with patch.object(darkmap, "open_tile", return_value=LazyArray(full)):
            median, info = darkmap.full_resolution_median("/data", "t", 4, 6)
        np.testing.assert_array_equal(median, np.full((2, 2), 4.0))
        self.assertEqual(info["full_resolution_planes"], [4, 5])
        self.assertIsNone(info["temporal_variance"])

    def test_stops_at_the_last_plane(self):
        """A deeper pyramid level rounds the ratio; reads stay in range."""
        full = np.ones((10, 2, 2), dtype=np.float32)
        with patch.object(darkmap, "open_tile", return_value=LazyArray(full)):
            _, info = darkmap.full_resolution_median("/data", "t", 2, 3)
        self.assertEqual(info["full_resolution_planes"], [6, 9])


class TestBuildDarkmap(unittest.TestCase):
    """The darkfield and its checks."""

    def _build(self, darkest=DARKEST, n_tiles=3):
        """
        Build a darkfield from the test tiles.

        Parameters
        ----------
        darkest : list of tuple of (int, float), optional
            Darkest plane index and level of each tile.
        n_tiles : int, optional
            Number of darkest tiles to use.

        Returns
        -------
        tuple of (np.ndarray, dict)
            The darkfield and its report.
        """
        with patch.object(
            darkmap, "open_tile", _opener(_tile_arrays(darkest))
        ):
            return darkmap.build_darkmap("/data", TILES, 3, n_tiles)

    def test_median_of_the_darkest_tiles(self):
        """Tiles 1, 3 and 0 (90, 91, 92 counts); the median is 91."""
        dark, report = self._build()
        self.assertEqual(dark.dtype, np.float32)
        np.testing.assert_array_equal(dark, _pattern(91.0))
        self.assertEqual(
            [group["tile"] for group in report["tiles_used"]],
            [TILES[1], TILES[3], TILES[0]],
        )
        self.assertEqual(report["tiles_used"][0]["coarse_plane"], 0)
        self.assertEqual(report["tiles_used"][0]["coarse_plane_mean"], 90.0)
        self.assertEqual(report["n_tiles_measured"], 5)
        self.assertEqual(report["n_tiles_used"], 3)

    def test_records_every_tile_and_the_spread(self):
        """The darkest level of each tile, for review."""
        _, report = self._build()
        self.assertEqual(report["darkest_plane_per_tile"][TILES[4]], 100.0)
        self.assertGreater(report["darkest_plane_spread"]["spread"], 3.0)

    def test_warns_when_the_dark_planes_contain_light(self):
        """Darkest planes from 90 to 100 counts: not a camera dark."""
        _, report = self._build()
        self.assertTrue(any("contain light" in w for w in report["warnings"]))

    def test_warns_when_the_darkfield_is_too_high(self):
        """With every tile used, the map (92) is above two tiles (90, 91)."""
        _, report = self._build(n_tiles=5)
        self.assertEqual(report["over_subtracted_tiles"], 0.4)
        self.assertTrue(
            any("removes real signal" in w for w in report["warnings"])
        )

    def test_tiles_within_the_tolerance_are_not_counted(self):
        """Darkest planes 0.03 counts under the map are noise, not light."""
        darkest = [(1, 90.47), (2, 90.5), (0, 90.5), (3, 90.47), (4, 90.5)]
        _, report = self._build(darkest)
        self.assertEqual(report["over_subtracted_tiles"], 0.0)

    def test_a_camera_dark_raises_no_warning(self):
        """Every tile's darkest plane at the same 90.5 counts."""
        darkest = [(1, 90.5), (2, 90.5), (0, 90.5), (3, 90.5), (4, 90.5)]
        dark, report = self._build(darkest)
        self.assertLess(report["darkest_plane_spread"]["spread"], 3.0)
        self.assertEqual(report["over_subtracted_tiles"], 0.0)
        self.assertEqual(report["warnings"], [])
        self.assertAlmostEqual(
            report["darkfield_stats"]["mean"], float(dark.mean()), places=5
        )


class TestMain(unittest.TestCase):
    """The entry point, end to end."""

    def test_writes_the_darkfield_and_its_sidecar(self):
        """The .npy for the estimator, a preview and the sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(darkmap, "list_tiles", return_value=TILES):
                with patch.object(
                    darkmap, "open_tile", _opener(_tile_arrays())
                ):
                    darkmap.main(
                        [
                            "s3://bucket/dataset/SPIM",
                            "--tile-pattern",
                            r"_ch_488\.ome\.zarr$",
                            "--output-name",
                            "ch_488",
                            "--n-tiles",
                            "3",
                            "--output-folder",
                            tmp,
                        ]
                    )
            written = sorted(path.name for path in Path(tmp).iterdir())
            dark = np.load(Path(tmp) / "darkfield_ch_488.npy")
            sidecar = json.loads(
                (Path(tmp) / "darkfield_ch_488.json").read_text()
            )
            log_written = (Path(tmp) / "metadata" / "darkmap.log").exists()
        self.assertEqual(
            written,
            [
                "darkfield_ch_488.json",
                "darkfield_ch_488.npy",
                "darkfield_ch_488.png",
                "metadata",
            ],
        )
        np.testing.assert_array_equal(dark, _pattern(91.0))
        self.assertEqual(sidecar["label"], "ch_488")
        self.assertEqual(sidecar["n_tiles_used"], 3)
        self.assertEqual(sidecar["provenance"]["args"]["n_tiles"], 3)
        self.assertTrue(log_written)

    def test_names_the_output_after_the_folder(self):
        """Without --output-name, the last path segment is used."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(darkmap, "list_tiles", return_value=TILES):
                with patch.object(
                    darkmap, "open_tile", _opener(_tile_arrays())
                ):
                    darkmap.main(["/data/ch_488", "--output-folder", tmp])
            self.assertTrue((Path(tmp) / "darkfield_ch_488.npy").exists())


if __name__ == "__main__":
    unittest.main()
