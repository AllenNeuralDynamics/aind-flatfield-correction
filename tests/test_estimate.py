"""Tests for the CLI and the per-run orchestration."""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aind_flatfield_correction.core.basicpy import estimate, fit, tiles
from aind_flatfield_correction.core.basicpy.config import (
    BASIC_CONFIG,
    baseline_params,
)
from aind_flatfield_correction.core.basicpy.tiles import (
    FitStack,
    TileRecord,
)
from tests.fakes import (
    InlineExecutor,
    LazyArray,
    inline_as_completed,
    make_basic,
)

TILE_NAMES = [
    f"Tile_X_{i:04d}_Y_0000_Z_0000_ch_405.ome.zarr" for i in range(4)
]


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


def _stack(n_tiles=4, planes=2, size=8):
    """
    Build a small fitting stack with index-keyed spans.

    Parameters
    ----------
    n_tiles : int, optional
        Number of tiles, by default 4.
    planes : int, optional
        Planes per tile, by default 2.
    size : int, optional
        Plane height and width, by default 8.

    Returns
    -------
    FitStack
        A stack of constant planes above a pedestal.
    """
    slices = np.full((n_tiles * planes, size, size), 190.0, dtype=np.float32)
    offsets = {
        index: (index * planes, (index + 1) * planes)
        for index in range(n_tiles)
    }
    manifest = tuple(
        TileRecord(
            name=TILE_NAMES[index],
            shape=(10, size, size),
            z_indices=tuple(range(planes)),
            mean=190.0,
            padded=False,
        )
        for index in range(n_tiles)
    )
    return FitStack(slices, offsets, planes, size, size, manifest)


def _args(**overrides):
    """
    Parse a default argument set with overrides applied.

    Parameters
    ----------
    **overrides : Any
        Attributes to set on the parsed namespace.

    Returns
    -------
    argparse.Namespace
        The parsed and patched arguments.
    """
    args = estimate.build_parser().parse_args(["/data/ch_405"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestDefaultLabel(unittest.TestCase):
    """Deriving the output name from the channel folder."""

    def test_uses_the_last_segment_of_an_s3_path(self):
        """The folder names the channel."""
        self.assertEqual(
            estimate.default_label("s3://bucket/DS/SPIM/ch_405"), "ch_405"
        )

    def test_ignores_a_trailing_slash(self):
        """Paths get pasted with and without one."""
        self.assertEqual(
            estimate.default_label("/data/SPIM/ch_488/"), "ch_488"
        )

    def test_replaces_filesystem_unsafe_characters(self):
        """The label becomes part of a filename."""
        self.assertEqual(
            estimate.default_label("/data/ch 488 (v2)"), "ch_488_v2"
        )

    def test_falls_back_when_there_is_no_segment(self):
        """A bare scheme has nothing to name the run after."""
        self.assertEqual(estimate.default_label("s3://"), "flatfield")


class TestBuildParser(unittest.TestCase):
    """The command-line surface."""

    def test_requires_only_a_base_path(self):
        """Everything else has a usable default."""
        args = estimate.build_parser().parse_args(["/data/ch_405"])
        self.assertEqual(args.base_path, "/data/ch_405")
        self.assertEqual(args.pyramid_level, 3)
        self.assertEqual(args.method, "fit")
        self.assertIsNone(args.tile_pattern)
        self.assertIsNone(args.output_name)

    def test_darkfield_value_parses_as_a_float(self):
        """It is subtracted from the image data, not concatenated."""
        args = estimate.build_parser().parse_args(
            ["/data", "--darkfield-value", "90"]
        )
        self.assertEqual(args.darkfield_value, 90.0)
        self.assertIsInstance(args.darkfield_value, float)

    def test_accepts_the_tile_filter_and_output_name(self):
        """Both replace the old per-channel grouping."""
        args = estimate.build_parser().parse_args(
            [
                "/data",
                "--tile-pattern",
                r"_ch_405\.ome\.zarr$",
                "--output-name",
                "ch405",
            ]
        )
        self.assertEqual(args.tile_pattern, r"_ch_405\.ome\.zarr$")
        self.assertEqual(args.output_name, "ch405")

    def test_rejects_an_unknown_method(self):
        """Only the two implemented estimators are offered."""
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                with self.assertRaises(SystemExit):
                    estimate.build_parser().parse_args(
                        ["/data", "--method", "magic"]
                    )


class TestSelectParameters(unittest.TestCase):
    """Choosing the smoothness for the final fit."""

    def test_skips_the_search_on_request(self):
        """--skip-search uses the known-good parameters directly."""
        params, report, confirm = estimate.select_parameters(
            np.ones((4, 4, 4), dtype=np.float32),
            {0: (0, 4)},
            _args(skip_search=True),
            BASIC_CONFIG,
        )
        self.assertEqual(params, baseline_params(BASIC_CONFIG))
        self.assertEqual(report["reason"], "skip_search_flag")
        self.assertIsNone(confirm)

    def test_skip_search_fits_a_supplied_smoothness(self):
        """--basic-config is how a hand-tuned value is provided."""
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        params, _, _ = estimate.select_parameters(
            np.ones((4, 4, 4), dtype=np.float32),
            {0: (0, 4)},
            _args(skip_search=True),
            config,
        )
        self.assertEqual(params, {"smoothness_flatfield": 2.5})

    def test_does_not_confirm_a_baseline_win(self):
        """There is nothing to confirm if the incumbent won."""
        report = {"chose_baseline": True}
        with patch.object(
            estimate,
            "parallel_autotune",
            return_value=(baseline_params(BASIC_CONFIG), report),
        ):
            with patch.object(
                estimate, "confirm_against_baseline"
            ) as confirmer:
                _, _, confirm = estimate.select_parameters(
                    np.ones((4, 4, 4), dtype=np.float32),
                    {0: (0, 4)},
                    _args(),
                    BASIC_CONFIG,
                )
        confirmer.assert_not_called()
        self.assertIsNone(confirm)

    def test_confirms_a_winner_at_scale(self):
        """A win on ~150 slices must hold up on thousands."""
        winner = {"smoothness_flatfield": 0.01}
        with patch.object(
            estimate,
            "parallel_autotune",
            return_value=(winner, {"chose_baseline": False}),
        ):
            with patch.object(
                estimate,
                "confirm_against_baseline",
                return_value=(winner, {"held_up": True}),
            ) as confirmer:
                params, _, confirm = estimate.select_parameters(
                    np.ones((4, 4, 4), dtype=np.float32),
                    {0: (0, 4)},
                    _args(),
                    BASIC_CONFIG,
                )
        confirmer.assert_called_once()
        self.assertEqual(params, winner)
        self.assertTrue(confirm["held_up"])

    def test_confirmation_can_be_disabled(self):
        """--n-confirm 0 skips the second opinion."""
        winner = {"smoothness_flatfield": 0.01}
        with patch.object(
            estimate,
            "parallel_autotune",
            return_value=(winner, {"chose_baseline": False}),
        ):
            with patch.object(
                estimate, "confirm_against_baseline"
            ) as confirmer:
                _, _, confirm = estimate.select_parameters(
                    np.ones((4, 4, 4), dtype=np.float32),
                    {0: (0, 4)},
                    _args(n_confirm=0),
                    BASIC_CONFIG,
                )
        confirmer.assert_not_called()
        self.assertIsNone(confirm)


class TestRunFit(unittest.TestCase):
    """Dispatching to the requested estimator."""

    def test_dispatches_to_the_joint_fit(self):
        """The default fits every gathered plane at once."""
        stack = _stack()
        with patch.object(
            estimate,
            "estimate_joint",
            return_value=(_plausible(), np.zeros((8, 8)), None),
        ) as joint:
            estimate._run_fit(stack.slices, stack, "fit", {})
        joint.assert_called_once()

    def test_dispatches_to_the_per_z_median(self):
        """The per-Z estimator needs the spans and the plane count."""
        stack = _stack()
        with patch.object(
            estimate,
            "estimate_per_z_median",
            return_value=(_plausible(), np.zeros((8, 8)), 2),
        ) as per_z:
            estimate._run_fit(stack.slices, stack, "per-z-median", {})
        per_z.assert_called_once_with(
            stack.slices, stack.z_offsets, stack.n_planes, {}
        )


class TestFitWithGuard(unittest.TestCase):
    """The final fit and its fallback."""

    def test_accepts_a_plausible_fit(self):
        """No refit when the guard is happy."""
        stack = _stack()
        with patch.object(
            estimate,
            "_run_fit",
            return_value=(_plausible(), np.zeros((8, 8)), None),
        ) as runner:
            result = estimate.fit_with_guard(
                stack.slices,
                stack,
                "fit",
                {"smoothness_flatfield": 0.5},
                None,
                BASIC_CONFIG,
            )
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(result.ok)
        self.assertEqual(result.params, {"smoothness_flatfield": 0.5})

    def test_refits_with_manual_params_after_a_bad_fit(self):
        """The known-good parameters are the safety net."""
        stack = _stack()
        outcomes = [
            (np.ones((8, 8)), np.zeros((8, 8)), None),
            (_plausible(), np.zeros((8, 8)), None),
        ]
        with patch.object(estimate, "_run_fit", side_effect=outcomes):
            result = estimate.fit_with_guard(
                stack.slices,
                stack,
                "fit",
                {"smoothness_flatfield": 0.5},
                None,
                BASIC_CONFIG,
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.params, baseline_params(BASIC_CONFIG))

    def test_refits_with_a_supplied_smoothness(self):
        """The fallback is the configured value, not the built-in 1.0."""
        stack = _stack()
        config = {**BASIC_CONFIG, "smoothness_flatfield": 2.5}
        outcomes = [
            (np.ones((8, 8)), np.zeros((8, 8)), None),
            (_plausible(), np.zeros((8, 8)), None),
        ]
        with patch.object(estimate, "_run_fit", side_effect=outcomes):
            result = estimate.fit_with_guard(
                stack.slices,
                stack,
                "fit",
                {"smoothness_flatfield": 0.5},
                None,
                config,
            )
        self.assertEqual(result.params, {"smoothness_flatfield": 2.5})

    def test_does_not_refit_what_the_manual_params_produced(self):
        """Refitting the same parameters would give the same field."""
        stack = _stack()
        with patch.object(
            estimate,
            "_run_fit",
            return_value=(np.ones((8, 8)), np.zeros((8, 8)), None),
        ) as runner:
            result = estimate.fit_with_guard(
                stack.slices,
                stack,
                "fit",
                baseline_params(BASIC_CONFIG),
                None,
                BASIC_CONFIG,
            )
        self.assertEqual(runner.call_count, 1)
        self.assertFalse(result.ok)

    def test_records_a_flatfield_that_stays_suspicious(self):
        """The sidecar has to say so, since nothing else will."""
        stack = _stack()
        bad = (np.ones((8, 8)), np.zeros((8, 8)), None)
        with patch.object(estimate, "_run_fit", side_effect=[bad, bad]):
            result = estimate.fit_with_guard(
                stack.slices,
                stack,
                "fit",
                {"smoothness_flatfield": 0.5},
                None,
                BASIC_CONFIG,
            )
        self.assertFalse(result.ok)
        self.assertTrue(result.reasons)


class TestWriteProducts(unittest.TestCase):
    """Saving the arrays and the figures."""

    def _result(self):
        """
        Build a passing fit result.

        Returns
        -------
        estimate.FitResult
            A plausible result with no guard complaints.
        """
        return estimate.FitResult(
            _plausible(),
            np.zeros((8, 8)),
            None,
            baseline_params(BASIC_CONFIG),
            True,
            {},
            [],
        )

    def test_saves_the_flatfield_and_the_pedestal(self):
        """The saved darkfield is the pedestal, not BaSiC's zeros."""
        stack = _stack()
        dark = np.full((8, 8), 90.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            estimate._write_products(
                "ch_405",
                out,
                self._result(),
                dark,
                stack,
                TILE_NAMES,
                _args(validate=False),
            )
            saved = np.load(out / "darkfield_ch_405.npy")
            self.assertTrue((out / "flatfield_ch_405.npy").exists())
        self.assertTrue(np.all(saved == 90.0))

    def test_skips_the_figures_without_validate(self):
        """Figures cost matplotlib and a lot of wall clock."""
        stack = _stack()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "save_validation_figures") as plotter:
                validation = estimate._write_products(
                    "ch_405",
                    Path(tmp),
                    self._result(),
                    np.zeros((8, 8), dtype=np.float32),
                    stack,
                    TILE_NAMES,
                    _args(validate=False),
                )
        plotter.assert_not_called()
        self.assertIsNone(validation)

    def test_writes_the_figures_with_validate(self):
        """--validate is the acceptance gate."""
        stack = _stack()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                estimate,
                "save_validation_figures",
                return_value={"cv_x_before": 9.0},
            ) as plotter:
                validation = estimate._write_products(
                    "ch_405",
                    Path(tmp),
                    self._result(),
                    np.zeros((8, 8), dtype=np.float32),
                    stack,
                    TILE_NAMES,
                    _args(validate=True, validate_tiles=2),
                )
        plotter.assert_called_once()
        self.assertEqual(validation["cv_x_before"], 9.0)


class TestEstimateDataset(unittest.TestCase):
    """One run, one flatfield."""

    def _run(self, tmp, **overrides):
        """
        Run the estimation with the loader and fit patched out.

        Parameters
        ----------
        tmp : str
            Output folder.
        **overrides : Any
            Argument overrides.

        Returns
        -------
        dict
            The sidecar that was written.
        """
        args = _args(
            skip_search=True,
            validate=False,
            darkfield_value=90.0,
            **overrides,
        )
        with patch.object(estimate, "load_fit_stack", return_value=_stack()):
            with patch.object(
                estimate,
                "_run_fit",
                return_value=(_plausible(), np.zeros((8, 8)), None),
            ):
                with patch.object(
                    estimate, "probe_plane_shape", return_value=(32, 32)
                ):
                    return estimate.estimate_dataset(
                        "ch_405",
                        TILE_NAMES,
                        90.0,
                        args,
                        Path(tmp),
                        dict(BASIC_CONFIG),
                        {"package_version": "1.0.0"},
                    )

    def test_writes_the_products(self):
        """Both fields, their full-resolution twins, and the sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            written = sorted(path.name for path in Path(tmp).iterdir())
        self.assertEqual(
            written,
            [
                "darkfield_ch_405.npy",
                "darkfield_ch_405_level0.npy",
                "flatfield_ch_405.json",
                "flatfield_ch_405.npy",
                "flatfield_ch_405_level0.npy",
                "metadata",
            ],
        )

    def test_upsamples_a_scalar_pedestal_to_a_constant_plane(self):
        """A scalar needs filling at the destination extent, not resizing."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            dark = np.load(Path(tmp) / "darkfield_ch_405_level0.npy")
        self.assertEqual(dark.shape, (32, 32))
        self.assertTrue(np.all(dark == 90.0))

    def test_upsampling_can_be_turned_off(self):
        """--no-upsample leaves only the estimation level."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp, upsample=False)
            written = [path.name for path in Path(tmp).iterdir()]
        self.assertNotIn("flatfield_ch_405_level0.npy", written)
        self.assertNotIn("darkfield_ch_405_level0.npy", written)
        self.assertIsNone(sidecar["upsampled"])

    def test_the_sidecar_records_the_upsampling(self):
        """Which level the full-resolution fields belong to."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
        upsampled = sidecar["upsampled"]
        self.assertEqual(upsampled["level"], "0")
        self.assertEqual(upsampled["shape"], [32, 32])
        self.assertEqual(upsampled["scale_factor"], 4.0)
        self.assertEqual(upsampled["probed_tile"], TILE_NAMES[0])
        self.assertTrue(
            upsampled["flatfield_path"].endswith("flatfield_ch_405_level0.npy")
        )
        self.assertTrue(
            upsampled["darkfield_path"].endswith("darkfield_ch_405_level0.npy")
        )

    def test_the_sidecar_records_the_provenance(self):
        """The tile names are not recoverable from anything else."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
            on_disk = json.loads(
                (Path(tmp) / "flatfield_ch_405.json").read_text()
            )
        self.assertEqual(sidecar["label"], "ch_405")
        self.assertEqual(on_disk["tiles"], TILE_NAMES)
        self.assertEqual(on_disk["n_tiles"], 4)
        self.assertEqual(on_disk["base_path"], "/data/ch_405")
        self.assertFalse(on_disk["suspicious"])

    def test_the_sidecar_records_the_darkfield_used(self):
        """A flatfield is only valid for the pedestal it was fitted on."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
        self.assertEqual(sidecar["darkfield_value"], 90.0)
        self.assertEqual(sidecar["darkfield_mean"], 90.0)
        self.assertEqual(sidecar["pedestal"]["median_raw"], 190.0)

    def test_writes_the_tile_manifest(self):
        """One record per tile, in the metadata folder."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
            manifest = json.loads(
                (Path(tmp) / "metadata" / "tiles_ch_405.json").read_text()
            )
        self.assertEqual(len(manifest), len(TILE_NAMES))
        self.assertEqual(
            sidecar["metadata_files"]["tile_manifest"],
            "metadata/tiles_ch_405.json",
        )

    def test_the_sidecar_summarizes_the_stack(self):
        """Extent, plane counts and intensity range of what was fitted."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
        summary = sidecar["stack"]
        self.assertEqual((summary["height"], summary["width"]), (8, 8))
        self.assertEqual(summary["n_images"], 8)
        self.assertEqual(summary["median"], 190.0)
        self.assertEqual(summary["n_padded_tiles"], 0)

    def test_the_sidecar_carries_the_provenance(self):
        """Passed in by main, recorded verbatim."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = self._run(tmp)
        self.assertEqual(sidecar["provenance"]["package_version"], "1.0.0")

    def test_the_sidecar_is_json_serializable(self):
        """numpy scalars would otherwise break the dump."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            json.loads((Path(tmp) / "flatfield_ch_405.json").read_text())


class TestUpsampleToFullResolution(unittest.TestCase):
    """Resampling both fields to the full-resolution extent."""

    def _upsample(self, tmp, dark):
        """
        Upsample against a probed extent of 32x32.

        Parameters
        ----------
        tmp : str
            Output folder.
        dark : np.ndarray or float
            Darkfield as supplied by the caller.

        Returns
        -------
        dict
            The upsampling record.
        """
        with patch.object(
            estimate, "probe_plane_shape", return_value=(32, 32)
        ):
            return estimate.upsample_to_full_resolution(
                "ch_405",
                Path(tmp),
                _plausible(),
                dark,
                TILE_NAMES[0],
                _args(),
            )

    def test_resamples_a_darkfield_image_from_the_original(self):
        """Not from the copy matched to the estimation level.

        A darkfield already at full resolution must come back
        bit-identical; going via the downsampled copy would blur it.
        """
        original = np.linspace(80.0, 100.0, 32 * 32, dtype=np.float32)
        original = original.reshape(32, 32)
        with tempfile.TemporaryDirectory() as tmp:
            self._upsample(tmp, original)
            written = np.load(Path(tmp) / "darkfield_ch_405_level0.npy")
        np.testing.assert_array_equal(written, original)

    def test_resamples_a_low_resolution_darkfield_image(self):
        """A darkfield captured at a coarser extent still scales up."""
        coarse = np.full((8, 8), 95.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            self._upsample(tmp, coarse)
            written = np.load(Path(tmp) / "darkfield_ch_405_level0.npy")
        self.assertEqual(written.shape, (32, 32))
        self.assertTrue(np.allclose(written, 95.0))

    def test_reports_the_scale_factor(self):
        """Read from the metadata, not assumed from the level number."""
        with tempfile.TemporaryDirectory() as tmp:
            record = self._upsample(tmp, 90.0)
        self.assertEqual(record["scale_factor"], 4.0)
        self.assertEqual(record["shape"], [32, 32])

    def test_preserves_the_flatfield_value_range(self):
        """Interpolating a gain field must not rescale it."""
        with tempfile.TemporaryDirectory() as tmp:
            self._upsample(tmp, 90.0)
            written = np.load(Path(tmp) / "flatfield_ch_405_level0.npy")
        self.assertAlmostEqual(float(written.min()), 0.75, places=5)
        self.assertAlmostEqual(float(written.max()), 1.25, places=5)


class TestMain(unittest.TestCase):
    """The entry point, end to end."""

    def _open_tile(self, base_path, name, level):
        """
        Serve a tile plane above a pedestal, sized by pyramid level.

        Answers any number of calls, which the shape probe, the plane
        load and the full-resolution probe all make. The full-resolution
        level is four times the estimation level, so the upsampling has
        something real to do.

        Parameters
        ----------
        base_path : str
            Ignored.
        name : str
            Ignored.
        level : int or str
            Multiscale level being opened.

        Returns
        -------
        LazyArray
            A ``(10, H, W)`` stack for that level.
        """
        size = 32 if str(level) == "0" else 8
        return LazyArray(np.full((10, size, size), 190.0, dtype=np.float32))

    def test_estimates_and_writes_a_flatfield(self):
        """From a folder of tiles to a saved flatfield and sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--darkfield-value",
                                "90",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            written = sorted(path.name for path in Path(tmp).iterdir())
        self.assertIn("flatfield_ch_405.npy", written)
        self.assertIn("flatfield_ch_405.json", written)

    def test_writes_the_upsampled_flatfield_by_default(self):
        """On without asking, at the extent the metadata reports."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            estimated = np.load(Path(tmp) / "flatfield_ch_405.npy")
            upsampled = np.load(Path(tmp) / "flatfield_ch_405_level0.npy")
        self.assertEqual(estimated.shape, (8, 8))
        self.assertEqual(upsampled.shape, (32, 32))

    def test_writes_the_run_log_and_manifest(self):
        """A Code Ocean run keeps its own diagnostics."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            metadata = Path(tmp) / "metadata"
            written = sorted(path.name for path in metadata.iterdir())
            sidecar = json.loads(
                (Path(tmp) / "flatfield_ch_405.json").read_text()
            )
        # The suite disables logging, so the log file is created but
        # empty; its contents are covered in tests/test_provenance.py.
        self.assertEqual(written, ["estimation.log", "tiles_ch_405.json"])
        self.assertEqual(
            sidecar["metadata_files"]["run_log"], "metadata/estimation.log"
        )
        self.assertIsNotNone(sidecar["provenance"]["command"])
        self.assertEqual(sidecar["provenance"]["args"]["max_fit_planes"], 8)

    def test_writes_the_figures_when_validating(self):
        """--validate produces the per-tile profile plots."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--validate",
                                "--validate-tiles",
                                "2",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            figure_dir = Path(tmp) / "ch_405"
            figures_written = sorted(
                path.name for path in figure_dir.iterdir()
            )
        self.assertEqual(len(figures_written), 4)

    def test_runs_the_per_z_median_method(self):
        """The alternative estimator, with the pool inline."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        with patch.object(
                            fit, "ProcessPoolExecutor", InlineExecutor
                        ):
                            with patch.object(
                                fit, "as_completed", inline_as_completed
                            ):
                                estimate.main(
                                    [
                                        "/data/ch_405",
                                        "--skip-search",
                                        "--method",
                                        "per-z-median",
                                        "--max-fit-planes",
                                        "8",
                                        "--output-folder",
                                        tmp,
                                    ]
                                )
            sidecar = json.loads(
                (Path(tmp) / "flatfield_ch_405.json").read_text()
            )
        self.assertEqual(sidecar["method"], "per-z-median")
        self.assertEqual(sidecar["n_z_fits_ok"], 2)

    def test_compiles_the_tile_pattern(self):
        """The filter reaches list_tiles as a compiled pattern."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                estimate, "list_tiles", return_value=TILE_NAMES
            ) as lister:
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--tile-pattern",
                                "_ch_405",
                                "--output-name",
                                "ch405",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            self.assertTrue((Path(tmp) / "flatfield_ch405.npy").exists())
        pattern = lister.call_args[0][1]
        self.assertEqual(pattern.pattern, "_ch_405")

    def test_passes_a_supplied_basic_config_to_the_fit(self):
        """The override reaches the solver and the sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    recorded = []
                    with patch.object(
                        fit, "BaSiC", make_basic(record=recorded)
                    ):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--basic-config",
                                '{"sort_intensity": false}',
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            sidecar = json.loads(
                (Path(tmp) / "flatfield_ch_405.json").read_text()
            )
        self.assertFalse(sidecar["basic_config"]["sort_intensity"])
        # The default that was not overridden must survive.
        self.assertEqual(sidecar["basic_config"]["fitting_mode"], "ladmap")
        self.assertFalse(recorded[-1]["sort_intensity"])

    def test_a_supplied_smoothness_reaches_the_fit_and_sidecar(self):
        """The answer to "how do I provide the smoothness"."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    recorded = []
                    with patch.object(
                        fit, "BaSiC", make_basic(record=recorded)
                    ):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--basic-config",
                                '{"smoothness_flatfield": 2.5}',
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                tmp,
                            ]
                        )
            sidecar = json.loads(
                (Path(tmp) / "flatfield_ch_405.json").read_text()
            )
        self.assertEqual(sidecar["params"]["smoothness_flatfield"], 2.5)
        self.assertEqual(sidecar["basic_config"]["smoothness_flatfield"], 2.5)
        self.assertEqual(recorded[-1]["smoothness_flatfield"], 2.5)

    def test_rejects_a_bad_basic_config_before_loading_tiles(self):
        """An unusable configuration must not cost a full tile load."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimate, "list_tiles") as lister:
                with self.assertRaises(FileNotFoundError):
                    estimate.main(
                        [
                            "/data/ch_405",
                            "--basic-config",
                            "/no/such/config.json",
                            "--output-folder",
                            tmp,
                        ]
                    )
        lister.assert_not_called()

    def test_creates_a_missing_output_folder(self):
        """The folder is named on the command line, not created first."""
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "results" / "flatfields"
            with patch.object(estimate, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        estimate.main(
                            [
                                "/data/ch_405",
                                "--skip-search",
                                "--max-fit-planes",
                                "8",
                                "--output-folder",
                                str(nested),
                            ]
                        )
            self.assertTrue(nested.is_dir())


if __name__ == "__main__":
    unittest.main()
