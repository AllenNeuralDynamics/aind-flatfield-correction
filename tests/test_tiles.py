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
    """Opening a tile as a 3-D array, across zarr versions."""

    def _reader(self, values=None):
        """
        Build a reader whose array has leading singleton axes.

        Parameters
        ----------
        values : np.ndarray, optional
            Array to serve; defaults to a 5-D ``(1, 1, 4, 8, 8)`` stack.

        Returns
        -------
        MagicMock
            A stand-in for an OMEZarrReader instance.
        """
        if values is None:
            values = np.zeros((1, 1, 4, 8, 8), dtype=np.float32)
        reader = MagicMock()
        reader.as_dask_array.return_value = LazyArray(values)
        return reader

    def test_drops_leading_singleton_axes(self):
        """OME-Zarr tiles arrive as (t, c, z, y, x)."""
        with patch.object(
            tiles, "OMEZarrReader", return_value=self._reader()
        ) as ctor:
            arr = tiles.open_tile("s3://bucket/ch/", CH405, 3)
        self.assertEqual(arr.shape, (4, 8, 8))
        ctor.assert_called_once_with(
            data_path=f"s3://bucket/ch/{CH405}",
            multiscale="3",
            zarr_version=tiles.ZARR_VERSIONS[0],
        )

    def test_falls_back_to_the_next_zarr_version(self):
        """Some datasets open only as v2 and others only as v3.

        The v2 reader surfaces a mismatched store as TypeError rather
        than ValueError, which once aborted the whole search on the
        first attempt.
        """
        with patch.object(
            tiles,
            "OMEZarrReader",
            side_effect=[TypeError("shape is None"), self._reader()],
        ) as ctor:
            arr = tiles.open_tile("/data", CH405, 3)
        self.assertEqual(arr.shape, (4, 8, 8))
        self.assertEqual(ctor.call_count, 2)
        self.assertEqual(
            ctor.call_args_list[1].kwargs["zarr_version"],
            tiles.ZARR_VERSIONS[1],
        )

    def test_raises_when_no_version_can_read_the_tile(self):
        """A tile no version can read is a real problem."""
        with patch.object(
            tiles,
            "OMEZarrReader",
            side_effect=TypeError("shape is None"),
        ) as ctor:
            with self.assertRaises(ValueError) as caught:
                tiles.open_tile("/data", CH405, 3)
        self.assertEqual(ctor.call_count, len(tiles.ZARR_VERSIONS))
        self.assertIn(CH405, str(caught.exception))

    def test_honours_an_explicit_zarr_version(self):
        """A caller who knows the format skips the search."""
        with patch.object(
            tiles, "OMEZarrReader", return_value=self._reader()
        ) as ctor:
            tiles.open_tile("/data", CH405, 0, zarr_version="3.0")
        ctor.assert_called_once_with(
            data_path=f"/data/{CH405}",
            multiscale="0",
            zarr_version="3.0",
        )

    def test_reports_a_failing_explicit_version(self):
        """No fallback when the version was asked for by name."""
        with patch.object(
            tiles, "OMEZarrReader", side_effect=TypeError("shape is None")
        ) as ctor:
            with self.assertRaises(ValueError) as caught:
                tiles.open_tile("/data", CH405, 3, zarr_version="2.0")
        self.assertEqual(ctor.call_count, 1)
        self.assertIn("2.0", str(caught.exception))
        # The original failure stays attached for diagnosis.
        self.assertIsInstance(caught.exception.__cause__, TypeError)


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

    def test_records_a_manifest_entry_per_tile(self):
        """The only record of which planes the fit was built from."""
        shapes = [(20, 8, 8)] * 3
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b", "c"], 3, 6)
        self.assertEqual(len(stack.manifest), 3)
        self.assertEqual(
            [record.name for record in stack.manifest], ["a", "b", "c"]
        )

    def test_the_manifest_records_the_planes_taken(self):
        """Which Z indices, not just how many."""
        shapes = [(20, 8, 8)]
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a"], 3, 2)
        record = stack.manifest[0]
        self.assertEqual(record.shape, (20, 8, 8))
        self.assertEqual(len(record.z_indices), stack.n_planes)
        self.assertTrue(all(4 <= z < 16 for z in record.z_indices))

    def test_the_manifest_flags_a_padded_tile(self):
        """A tile smaller than the rest leaves zeros in the stack."""
        shapes = [(20, 8, 8), (20, 6, 6)]
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b"], 3, 4)
        self.assertFalse(stack.manifest[0].padded)
        self.assertTrue(stack.manifest[1].padded)

    def test_the_manifest_records_each_tile_mean(self):
        """A dead tile shows up as a mean near zero."""
        shapes = [(20, 8, 8), (20, 8, 8)]
        with patch.object(
            tiles, "open_tile", side_effect=self._arrays(shapes) * 2
        ):
            stack = tiles.load_fit_stack("/data", ["a", "b"], 3, 4)
        self.assertEqual(stack.manifest[0].mean, 1.0)
        self.assertEqual(stack.manifest[1].mean, 2.0)


class TestResizePlane(unittest.TestCase):
    """Resampling a 2-D field onto another extent."""

    def test_upsamples_without_changing_the_value_range(self):
        """preserve_range: these are gains, not [0, 1] intensities."""
        plane = np.linspace(0.75, 1.25, 16).reshape(4, 4)
        resized = tiles.resize_plane(plane, (16, 16))
        self.assertEqual(resized.shape, (16, 16))
        self.assertAlmostEqual(float(resized.min()), 0.75, places=5)
        self.assertAlmostEqual(float(resized.max()), 1.25, places=5)

    def test_downsamples_too(self):
        """The same helper matches a darkfield to the fitting level."""
        plane = np.linspace(0.0, 1.0, 256).reshape(16, 16)
        self.assertEqual(tiles.resize_plane(plane, (4, 4)).shape, (4, 4))

    def test_returns_the_input_when_the_shape_matches(self):
        """No interpolation, and no needless copy."""
        plane = np.ones((4, 4), dtype=np.float32)
        self.assertIs(tiles.resize_plane(plane, (4, 4)), plane)

    def test_always_returns_float32(self):
        """Downstream saves and divisions expect it."""
        plane = np.ones((4, 4), dtype=np.uint16)
        self.assertEqual(tiles.resize_plane(plane, (8, 8)).dtype, np.float32)


class TestProbePlaneShape(unittest.TestCase):
    """Reading the destination extent from the dataset metadata."""

    def test_reads_the_full_resolution_extent(self):
        """The scale factor between levels is not assumed."""
        arr = LazyArray(np.zeros((10, 64, 48), dtype=np.float32))
        with patch.object(tiles, "open_tile", return_value=arr) as opener:
            shape = tiles.probe_plane_shape("/data", CH405)
        self.assertEqual(shape, (64, 48))
        opener.assert_called_once_with("/data", CH405, "0")

    def test_ignores_leading_axes(self):
        """A tile may still carry its time and channel axes."""
        arr = LazyArray(np.zeros((1, 1, 10, 32, 32), dtype=np.float32))
        with patch.object(tiles, "open_tile", return_value=arr):
            self.assertEqual(tiles.probe_plane_shape("/data", CH405), (32, 32))

    def test_accepts_another_level(self):
        """Callers can probe the level they are estimating on."""
        arr = LazyArray(np.zeros((10, 8, 8), dtype=np.float32))
        with patch.object(tiles, "open_tile", return_value=arr) as opener:
            tiles.probe_plane_shape("/data", CH405, 3)
        opener.assert_called_once_with("/data", CH405, 3)


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


class TestPlaneConcentration(unittest.TestCase):
    """Measuring how unevenly the fit objective is spread across planes."""

    def test_a_uniform_stack_spreads_its_objective_evenly(self):
        """With every plane alike, the brightest few hold their fair share."""
        slices = np.ones((280, 4, 5), dtype=np.float32)
        info = tiles.plane_concentration(slices)
        self.assertEqual(info["top_n"], 10)
        self.assertEqual(info["n_planes"], 280)
        self.assertAlmostEqual(info["top_share"], 10 / 280, places=6)
        self.assertEqual(info["hot_planes"], 0)

    def test_a_few_bright_planes_dominate(self):
        """This is the injection-site signature the threshold keys on."""
        slices = np.ones((280, 4, 5), dtype=np.float32)
        slices[:10] *= 60.0
        info = tiles.plane_concentration(slices)
        self.assertGreater(info["top_share"], 0.5)
        self.assertEqual(info["hot_planes"], 10)
        self.assertAlmostEqual(info["plane_mean_ratio"], 60.0, places=4)

    def test_hot_planes_are_counted_separately_from_the_share(self):
        """Many mildly bright planes are not the same as a few dominant."""
        slices = np.ones((280, 4, 5), dtype=np.float32)
        slices[:44] *= 6.0
        info = tiles.plane_concentration(slices)
        self.assertEqual(info["hot_planes"], 44)
        # 44 hot planes, yet the brightest 10 hold well under half.
        self.assertLess(info["top_share"], 0.5)

    def test_an_all_zero_stack_reports_nan_rather_than_dividing(self):
        """A dead channel must not raise here."""
        info = tiles.plane_concentration(np.zeros((8, 2, 2), dtype=np.float32))
        self.assertTrue(np.isnan(info["top_share"]))
        self.assertTrue(np.isnan(info["plane_mean_ratio"]))

    def test_the_top_slice_is_at_least_one_plane(self):
        """A tiny stack still has a brightest plane."""
        info = tiles.plane_concentration(np.ones((3, 2, 2), dtype=np.float32))
        self.assertEqual(info["top_n"], 1)
        self.assertAlmostEqual(info["hot_fraction"], 0.0, places=6)


DIM, BRIGHT, PATCH = 10.0, 100.0, 1000.0


def _plane(kind, height=40, width=50):
    """
    Build one pedestal-subtracted test plane.

    Parameters
    ----------
    kind : str
        ``dim`` (uniform, dim), ``bright`` (uniform, bright),
        ``concentrated`` (dim with one very bright region: bright overall)
        or ``dim_patch`` (dim with one small, moderately bright region).
    height : int, optional
        Plane height, by default 40.
    width : int, optional
        Plane width, by default 50.

    Returns
    -------
    np.ndarray
        The plane.
    """
    plane = np.full((height, width), DIM, dtype=np.float32)
    if kind == "bright":
        plane[:] = BRIGHT
    elif kind == "concentrated":
        plane[5:20, 5:20] = PATCH
    elif kind == "dim_patch":
        plane[5:10, 5:10] = BRIGHT
    return plane


def _concentrated_stack(layout, pedestal=90.0):
    """
    Build a fitting stack and its raw stack from per-tile plane kinds.

    Parameters
    ----------
    layout : list of list of str
        Plane kinds of each tile, in Z order (see :func:`_plane`).
    pedestal : float, optional
        Value added to make the raw planes, by default 90.

    Returns
    -------
    tuple of (np.ndarray, FitStack)
        The pedestal-subtracted planes and the raw stack.
    """
    planes, offsets, manifest = [], {}, []
    for index, kinds in enumerate(layout):
        offsets[index] = (len(planes), len(planes) + len(kinds))
        planes.extend(_plane(kind) for kind in kinds)
        manifest.append(
            tiles.TileRecord(
                name=f"tile_{index}",
                shape=(100, 40, 50),
                z_indices=tuple(range(10, 10 + 2 * len(kinds), 2)),
                mean=0.0,
                padded=False,
            )
        )
    fit_slices = np.stack(planes)
    stack = tiles.FitStack(
        fit_slices + pedestal,
        offsets,
        len(layout[0]),
        40,
        50,
        tuple(manifest),
    )
    return fit_slices, stack


LAYOUT = [
    ["dim", "dim", "concentrated", "dim", "bright", "dim"],
    ["dim"] * 6,
    ["concentrated"] * 6,
    ["dim", "dim_patch", "dim", "dim", "dim", "dim"],
]


class TestBackgroundMap(unittest.TestCase):
    """The diffuse background of one plane on a coarse grid."""

    def test_blocks_come_from_the_short_side(self):
        """The same physical block at every pyramid level."""
        self.assertEqual(
            tiles.background_map(np.ones((400, 500))).shape, (40, 50)
        )
        self.assertEqual(
            tiles.background_map(np.ones((40, 50))).shape, (40, 50)
        )

    def test_ignores_sparse_bright_pixels(self):
        """A block median is not moved by a few bright cells."""
        plane = np.full((400, 500), 10.0, dtype=np.float32)
        plane[::10, ::10] = 1e4
        np.testing.assert_allclose(tiles.background_map(plane), 10.0)


class TestPlaneScores(unittest.TestCase):
    """Brightness and concentration of every plane."""

    def test_diffuse_and_concentrated_planes(self):
        """Uniform signal scores 1; one dominant region scores much more."""
        planes = np.stack(
            [_plane("dim"), _plane("concentrated"), np.zeros((40, 50))]
        )
        scores = tiles.plane_scores(planes)
        self.assertAlmostEqual(scores["localized"][0], 1.0, places=5)
        self.assertGreater(scores["localized"][1], 3.0)
        self.assertAlmostEqual(scores["level"][0], DIM, places=4)
        self.assertAlmostEqual(scores["mean"][0], DIM, places=4)

    def test_an_empty_plane_has_no_ratio(self):
        """Its median is zero, so it can never count as concentrated."""
        scores = tiles.plane_scores(np.zeros((1, 40, 50), dtype=np.float32))
        self.assertTrue(np.isnan(scores["localized"][0]))
        self.assertEqual(scores["level"][0], 0.0)


class TestConcentratedPlanes(unittest.TestCase):
    """Bright AND concentrated, never one of the two alone."""

    def test_needs_both_conditions(self):
        """Bright diffuse and dim concentrated planes are kept."""
        fit_slices, _ = _concentrated_stack(LAYOUT)
        bright, both = tiles.concentrated_planes(
            tiles.plane_scores(fit_slices)
        )
        kinds = [kind for tile in LAYOUT for kind in tile]
        self.assertEqual(
            [k for k, b in zip(kinds, bright) if b],
            ["concentrated", "bright"] + ["concentrated"] * 6,
        )
        self.assertEqual(
            [k for k, d in zip(kinds, both) if d], ["concentrated"] * 7
        )

    def test_nothing_is_bright_when_most_planes_are_empty(self):
        """A median plane mean of zero gives no scale for brightness."""
        scores = {
            "mean": np.array([0.0, 0.0, 5.0]),
            "localized": np.array([np.nan, np.nan, 10.0]),
        }
        bright, both = tiles.concentrated_planes(scores)
        self.assertFalse(bright.any())
        self.assertFalse(both.any())


class TestDropConcentratedPlanes(unittest.TestCase):
    """Removing the planes that teach the flatfield the tissue."""

    def test_drops_only_the_bright_concentrated_planes(self):
        """Seven planes go; the bright diffuse plane stays."""
        fit_slices, stack = _concentrated_stack(LAYOUT)
        kept, new_stack, report = tiles.drop_concentrated_planes(
            fit_slices, stack
        )
        self.assertEqual(report["n_candidates"], 24)
        self.assertEqual(report["n_dropped"], 7)
        self.assertEqual(report["n_kept"], 17)
        self.assertEqual(kept.shape, (17, 40, 50))
        self.assertEqual(new_stack.slices.shape, (17, 40, 50))
        np.testing.assert_allclose(new_stack.slices, kept + 90.0)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["rule"]["bright_ratio"], 3.0)
        self.assertEqual(
            report["top_share_before"],
            tiles.plane_concentration(fit_slices)["top_share"],
        )
        self.assertEqual(
            report["top_share_after"],
            tiles.plane_concentration(kept)["top_share"],
        )

    def test_rebuilds_the_spans_and_the_manifest(self):
        """A tile left with no planes keeps an empty span."""
        fit_slices, stack = _concentrated_stack(LAYOUT)
        _, new_stack, report = tiles.drop_concentrated_planes(
            fit_slices, stack
        )
        self.assertEqual(
            new_stack.z_offsets,
            {0: (0, 5), 1: (5, 11), 2: (11, 11), 3: (11, 17)},
        )
        self.assertEqual(new_stack.manifest[0].z_indices, (10, 12, 16, 18, 20))
        self.assertEqual(new_stack.manifest[2].z_indices, ())
        self.assertEqual(
            report["dropped_per_tile"], {"tile_0": 1, "tile_2": 6}
        )
        self.assertEqual(new_stack.n_planes, stack.n_planes)

    def test_reuses_given_scores(self):
        """The scores of the pre-fit checks are not computed twice."""
        fit_slices, stack = _concentrated_stack(LAYOUT)
        scores = tiles.plane_scores(fit_slices)
        with patch.object(tiles, "plane_scores") as scorer:
            tiles.drop_concentrated_planes(fit_slices, stack, scores)
        scorer.assert_not_called()

    def test_a_stack_without_a_manifest(self):
        """Spans are still rebuilt; nothing is named per tile."""
        fit_slices, stack = _concentrated_stack(LAYOUT)
        stack = stack._replace(manifest=())
        _, new_stack, report = tiles.drop_concentrated_planes(
            fit_slices, stack
        )
        self.assertEqual(new_stack.manifest, ())
        self.assertEqual(report["dropped_per_tile"], {})
        self.assertEqual(new_stack.z_offsets[2], (11, 11))


if __name__ == "__main__":
    unittest.main()
