"""Tests for tile discovery, plane loading and darkfield handling."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from aind_flatfield_correction.core.basicpy import tiles
from tests.fakes import LazyArray

CH405 = "Tile_X_0000_Y_0000_Z_0000_ch_405.ome.zarr"
CH405_B = "Tile_X_0001_Y_0000_Z_0000_ch_405.ome.zarr"
CH488 = "Tile_X_0000_Y_0000_Z_0000_ch_488.ome.zarr"


class TestListTileNamesS3(unittest.TestCase):
    """Listing an s3:// channel folder."""

    def test_lists_the_immediate_child_prefixes(self):
        """Only the folder's own entries, not the whole recursive tree."""
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {"CommonPrefixes": [{"Prefix": "DS/SPIM/ch_405/tile_a/"}]},
            {"CommonPrefixes": [{"Prefix": "DS/SPIM/ch_405/tile_b/"}]},
        ]
        with patch.object(tiles.boto3, "client", return_value=client):
            names = tiles._list_tile_names_s3("s3://bucket/DS/SPIM/ch_405")
        self.assertEqual(names, ["tile_a", "tile_b"])

    def test_splits_the_bucket_from_the_prefix(self):
        """A trailing delimiter is required or S3 returns the parent."""
        client = MagicMock()
        paginate = client.get_paginator.return_value.paginate
        paginate.return_value = []
        with patch.object(tiles.boto3, "client", return_value=client):
            tiles._list_tile_names_s3("s3://my-bucket/a/b")
        paginate.assert_called_once_with(
            Bucket="my-bucket", Prefix="a/b/", Delimiter="/"
        )

    def test_tolerates_a_page_without_prefixes(self):
        """An empty folder yields pages with no CommonPrefixes key."""
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{}]
        with patch.object(tiles.boto3, "client", return_value=client):
            self.assertEqual(
                tiles._list_tile_names_s3("s3://bucket/prefix"), []
            )


class TestListTileNamesLocal(unittest.TestCase):
    """Listing a local channel folder."""

    def test_returns_sorted_entries(self):
        """Order must be stable so tile indices are reproducible."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in (CH488, CH405):
                (Path(tmp) / name).mkdir()
            self.assertEqual(tiles._list_tile_names_local(tmp), [CH405, CH488])

    def test_rejects_a_path_that_is_not_a_directory(self):
        """A typo'd path should fail loudly, not silently list nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope")
            with self.assertRaises(FileNotFoundError):
                tiles._list_tile_names_local(missing)


class TestListTiles(unittest.TestCase):
    """Tile selection, with and without a filter pattern."""

    def setUp(self):
        """Patch the local lister so no filesystem is needed."""
        patcher = patch.object(
            tiles,
            "_list_tile_names_local",
            return_value=[CH405, CH405_B, CH488, "OME"],
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_takes_every_entry_without_a_pattern(self):
        """The folder is trusted to hold only tiles."""
        self.assertEqual(len(tiles.list_tiles("/data/ch_405")), 4)

    def test_filters_on_a_full_pattern(self):
        """An anchored pattern keeps only the matching channel."""
        kept = tiles.list_tiles("/data", re.compile(r"_ch_405\.ome\.zarr$"))
        self.assertEqual(kept, [CH405, CH405_B])

    def test_filters_on_a_fragment(self):
        """search(), not match(): a fragment is a usable filter."""
        kept = tiles.list_tiles("/data", re.compile("_ch_488"))
        self.assertEqual(kept, [CH488])

    def test_raises_when_a_pattern_matches_nothing(self):
        """The message names the pattern so the typo is findable."""
        with self.assertRaises(ValueError) as caught:
            tiles.list_tiles("/data", re.compile("nomatch"))
        self.assertIn("nomatch", str(caught.exception))

    def test_raises_on_an_empty_folder(self):
        """An empty folder is an error, not an empty fit."""
        with patch.object(tiles, "_list_tile_names_local", return_value=[]):
            with self.assertRaises(ValueError):
                tiles.list_tiles("/data")

    def test_uses_the_s3_lister_for_s3_paths(self):
        """The scheme picks the listing backend."""
        with patch.object(
            tiles, "_list_tile_names_s3", return_value=[CH405]
        ) as lister:
            self.assertEqual(tiles.list_tiles("s3://bucket/ch_405"), [CH405])
        lister.assert_called_once_with("s3://bucket/ch_405")


class TestOpenTile(unittest.TestCase):
    """Opening a tile as a 3-D array."""

    def test_drops_leading_singleton_axes(self):
        """OME-Zarr tiles arrive as (t, c, z, y, x)."""
        values = np.zeros((1, 1, 4, 8, 8), dtype=np.float32)
        reader = MagicMock()
        reader.as_dask_array.return_value = LazyArray(values)
        with patch.object(tiles, "OMEZarrReader", return_value=reader) as ctor:
            arr = tiles.open_tile("s3://bucket/ch/", CH405, 3)
        self.assertEqual(arr.shape, (4, 8, 8))
        ctor.assert_called_once_with(
            data_path=f"s3://bucket/ch/{CH405}",
            multiscale="3",
            zarr_version="3.0",
        )


class TestPickZPlanes(unittest.TestCase):
    """Choosing which Z planes to fit."""

    def test_takes_the_middle_sixty_percent(self):
        """Planes near either end are background-dominated."""
        planes = tiles.pick_z_planes(100, 5)
        self.assertGreaterEqual(planes.min(), 20)
        self.assertLess(planes.max(), 80)
        self.assertEqual(len(planes), 5)

    def test_clamps_to_the_available_window(self):
        """Asking for more planes than the window holds."""
        planes = tiles.pick_z_planes(10, 50)
        self.assertLessEqual(len(planes), 6)

    def test_handles_a_thin_stack(self):
        """A one-plane tile has no middle to speak of."""
        self.assertEqual(tiles.pick_z_planes(1, 3).tolist(), [0])

    def test_falls_back_when_the_window_collapses(self):
        """A wide edge fraction can leave no planes at all."""
        self.assertEqual(
            tiles.pick_z_planes(2, 2, edge_fraction=0.5).tolist(), [0, 1]
        )

    def test_edge_fraction_of_zero_uses_the_whole_stack(self):
        """Callers can opt out of the window."""
        planes = tiles.pick_z_planes(10, 10, edge_fraction=0.0)
        self.assertEqual(planes.tolist(), list(range(10)))


class TestProbeTileShapes(unittest.TestCase):
    """Reading tile shapes without fetching voxels."""

    def test_keys_by_tile_name_not_by_shape(self):
        """Keying by shape collapsed same-extent tiles into one entry."""
        names = [f"tile_{i}" for i in range(26)]
        shapes = [(4 + i % 2, 8, 8) for i in range(26)]
        arrays = [LazyArray(np.zeros(s, dtype=np.float32)) for s in shapes]
        with patch.object(tiles, "open_tile", side_effect=arrays):
            probed = tiles.probe_tile_shapes("/data", names, 3)
        self.assertEqual(len(probed), 26)
        self.assertEqual(min(s[0] for s in probed.values()), 4)


class TestLoadFitStack(unittest.TestCase):
    """Streaming planes from every tile into one stack."""

    def _arrays(self, shapes):
        """
        Build lazy arrays whose values identify their tile.

        Parameters
        ----------
        shapes : list of tuple
            Per-tile ``(Z, H, W)`` shapes.

        Returns
        -------
        list of LazyArray
            One array per shape, filled with its tile index.
        """
        return [
            LazyArray(np.full(shape, index + 1, dtype=np.float32))
            for index, shape in enumerate(shapes)
        ]

    def test_spends_the_budget_on_tiles_not_on_z(self):
        """A budget of 12 over 12 tiles is one plane each."""
        shapes = [(20, 8, 8)] * 12
        with patch.object(
            tiles,
            "open_tile",
            side_effect=self._arrays(shapes) * 2,
        ):
            stack = tiles.load_fit_stack("/data", ["t"] * 12, 3, 12)
        self.assertEqual(stack.n_planes, 1)
        self.assertEqual(stack.slices.shape, (12, 8, 8))

    def test_spans_are_keyed_by_tile_index(self):
        """Nothing is inferred from the tile name any more."""
        shapes = [(20, 8, 8)] * 3
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b", "c"], 3, 6)
        self.assertEqual(sorted(stack.z_offsets), [0, 1, 2])
        self.assertEqual(stack.z_offsets[0], (0, 2))
        self.assertEqual(stack.z_offsets[2], (4, 6))

    def test_pads_tiles_smaller_than_the_largest(self):
        """The stack is allocated at the largest plane extent."""
        shapes = [(20, 8, 8), (20, 6, 6)]
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b"], 3, 4)
        self.assertEqual((stack.height, stack.width), (8, 8))
        second = stack.slices[stack.z_offsets[1][0]]
        self.assertEqual(second[0, 0], 2.0)
        self.assertEqual(second[7, 7], 0.0)  # padding

    def test_planes_per_tile_is_capped_by_the_shortest_tile(self):
        """A short tile must not be over-sampled."""
        shapes = [(40, 4, 4), (3, 4, 4)]
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b"], 3, 40)
        self.assertLessEqual(stack.n_planes, 3)


class TestLoadDarkfield(unittest.TestCase):
    """Resolving the darkfield from the command line."""

    def test_scalar_value_when_no_image_is_given(self):
        """The common case is a single ADU pedestal."""
        self.assertEqual(tiles.load_darkfield(None, 90), 90.0)

    def test_loads_a_numpy_image(self):
        """A .npy darkfield is loaded as float32."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dark.npy")
            np.save(path, np.full((4, 4), 95, dtype=np.uint16))
            dark = tiles.load_darkfield(path, 0)
        self.assertEqual(dark.shape, (4, 4))
        self.assertEqual(dark.dtype, np.float32)

    def test_loads_a_tif_image_and_squeezes_it(self):
        """Camera darkfields are often 3-D tif stacks."""
        volume = np.full((1, 4, 4), 95, dtype=np.uint16)
        with patch.object(tiles, "tif_imread", return_value=volume) as reader:
            dark = tiles.load_darkfield("/tmp/dark.tif", 0)
        reader.assert_called_once_with("/tmp/dark.tif")
        self.assertEqual(dark.shape, (4, 4))

    def test_reraises_a_failed_load(self):
        """A bad path must not silently fall back to a scalar."""
        with patch.object(tiles, "tif_imread", side_effect=OSError("bad tif")):
            with self.assertRaises(OSError):
                tiles.load_darkfield("/tmp/dark.tiff", 0)


class TestMatchDarkfield(unittest.TestCase):
    """Broadcasting the darkfield onto the estimation plane."""

    def test_fills_a_plane_from_a_scalar(self):
        """A scalar pedestal becomes a constant plane."""
        plane = tiles.match_darkfield(90.0, (2, 3))
        self.assertEqual(plane.shape, (2, 3))
        self.assertTrue(np.all(plane == 90.0))

    def test_keeps_a_matching_image_untouched(self):
        """No resize when the shapes already agree."""
        dark = np.full((4, 4), 7.0, dtype=np.float32)
        plane = tiles.match_darkfield(dark, (4, 4))
        np.testing.assert_array_equal(plane, dark)

    def test_resizes_a_full_resolution_image(self):
        """A level-0 darkfield against a downsampled estimation level."""
        dark = np.linspace(0, 100, 64).reshape(8, 8)
        plane = tiles.match_darkfield(dark, (4, 4))
        self.assertEqual(plane.shape, (4, 4))
        self.assertEqual(plane.dtype, np.float32)


class TestSubtractPedestal(unittest.TestCase):
    """Removing the sensor offset before fitting."""

    def test_subtracts_and_clips_at_zero(self):
        """Below-pedestal pixels become zero, never negative."""
        slices = np.array([[[80.0, 100.0]]], dtype=np.float32)
        dark = np.full((1, 2), 90.0, dtype=np.float32)
        fitted, _ = tiles.subtract_pedestal(slices, dark)
        np.testing.assert_array_equal(fitted[0, 0], [0.0, 10.0])

    def test_reports_the_dilution_avoided(self):
        """Fitting on raw would have squashed the vignetting ~14x."""
        slices = np.full((2, 2, 2), 97.0, dtype=np.float32)
        dark = np.full((2, 2), 90.0, dtype=np.float32)
        _, info = tiles.subtract_pedestal(slices, dark)
        self.assertEqual(info["median_raw"], 97.0)
        self.assertEqual(info["median_pedestal_removed"], 7.0)
        self.assertAlmostEqual(
            info["dilution_if_fitted_on_raw"], 7.0 / 97.0, places=6
        )

    def test_an_all_zero_stack_reports_nan_dilution(self):
        """No division by a zero median."""
        slices = np.zeros((1, 2, 2), dtype=np.float32)
        dark = np.zeros((2, 2), dtype=np.float32)
        _, info = tiles.subtract_pedestal(slices, dark)
        self.assertTrue(np.isnan(info["dilution_if_fitted_on_raw"]))


if __name__ == "__main__":
    unittest.main()
