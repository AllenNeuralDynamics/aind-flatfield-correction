"""Tests for the command-line entry point."""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aind_flatfield_correction.core.basicpy import cli, fit, tiles
from tests.fakes import (
    InlineExecutor,
    LazyArray,
    inline_as_completed,
    make_basic,
)

TILE_NAMES = [
    f"Tile_X_{i:04d}_Y_0000_Z_0000_ch_405.ome.zarr" for i in range(4)
]


class TestBuildParser(unittest.TestCase):
    """The command-line surface."""

    def test_requires_only_a_base_path(self):
        """Everything else has a usable default."""
        args = cli.build_parser().parse_args(["/data/ch_405"])
        self.assertEqual(args.base_path, "/data/ch_405")
        self.assertEqual(args.pyramid_level, 3)
        self.assertEqual(args.method, "fit")
        self.assertIsNone(args.tile_pattern)
        self.assertIsNone(args.output_name)

    def test_darkfield_value_parses_as_a_float(self):
        """It is subtracted from the image data, not concatenated."""
        args = cli.build_parser().parse_args(
            ["/data", "--darkfield-value", "90"]
        )
        self.assertEqual(args.darkfield_value, 90.0)
        self.assertIsInstance(args.darkfield_value, float)

    def test_accepts_the_tile_filter_and_output_name(self):
        """Both replace the old per-channel grouping."""
        args = cli.build_parser().parse_args(
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
                    cli.build_parser().parse_args(
                        ["/data", "--method", "magic"]
                    )

    def test_scales_the_flatfield_by_default(self):
        """The fix for signal compression is on unless turned off."""
        args = cli.build_parser().parse_args(["/data/ch_405"])
        self.assertTrue(args.scale_flatfield)
        self.assertEqual(args.scale_percentile, 99.9)

    def test_scaling_can_be_turned_off_or_retargeted(self):
        """--no-scale-flatfield keeps the mean-1 field."""
        args = cli.build_parser().parse_args(
            ["/data", "--no-scale-flatfield", "--scale-percentile", "99"]
        )
        self.assertFalse(args.scale_flatfield)
        self.assertEqual(args.scale_percentile, 99.0)

    def test_the_plane_filter_is_opt_in(self):
        """Off unless asked for; it changes which planes are fitted."""
        args = cli.build_parser().parse_args(["/data/ch_405"])
        self.assertFalse(args.drop_concentrated_planes)
        args = cli.build_parser().parse_args(
            ["/data/ch_405", "--drop-concentrated-planes"]
        )
        self.assertTrue(args.drop_concentrated_planes)

    def test_rejects_an_unusable_percentile(self):
        """Caught at parse time, not after the fit."""
        for value in ("0", "100.5", "-1", "high"):
            with self.subTest(value=value):
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stderr(devnull):
                        with self.assertRaises(SystemExit):
                            cli.build_parser().parse_args(
                                ["/data", "--scale-percentile", value]
                            )


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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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

    def test_rejects_the_plane_filter_with_the_per_z_method(self):
        """Dropping single planes would leave the per-Z groups uneven."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stderr(devnull):
                    with patch.object(cli, "list_tiles") as lister:
                        with self.assertRaises(SystemExit):
                            cli.main(
                                [
                                    "/data/ch_405",
                                    "--method",
                                    "per-z-median",
                                    "--drop-concentrated-planes",
                                    "--output-folder",
                                    tmp,
                                ]
                            )
        lister.assert_not_called()

    def test_writes_the_upsampled_flatfield_by_default(self):
        """On without asking, at the extent the metadata reports."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
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
                                cli.main(
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
                cli, "list_tiles", return_value=TILE_NAMES
            ) as lister:
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    recorded = []
                    with patch.object(
                        fit, "BaSiC", make_basic(record=recorded)
                    ):
                        cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    recorded = []
                    with patch.object(
                        fit, "BaSiC", make_basic(record=recorded)
                    ):
                        cli.main(
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
            with patch.object(cli, "list_tiles") as lister:
                with self.assertRaises(FileNotFoundError):
                    cli.main(
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
            with patch.object(cli, "list_tiles", return_value=TILE_NAMES):
                with patch.object(
                    tiles, "open_tile", side_effect=self._open_tile
                ):
                    with patch.object(fit, "BaSiC", make_basic()):
                        cli.main(
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
