"""Tests for run provenance and log capture."""

import argparse
import logging
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

from aind_flatfield_correction import provenance

STARTED = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
FINISHED = STARTED + timedelta(seconds=90)


def _args(**overrides):
    """
    Build a namespace standing in for parsed arguments.

    Parameters
    ----------
    **overrides : Any
        Attributes to set.

    Returns
    -------
    argparse.Namespace
        The namespace.
    """
    defaults = {
        "base_path": "/data/ch_405",
        "pyramid_level": 3,
        "darkfield_image": None,
        "darkfield_value": 90.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestUtcNow(unittest.TestCase):
    """The run's clock."""

    def test_is_timezone_aware(self):
        """A naive timestamp is ambiguous in a shared results folder."""
        self.assertIsNotNone(provenance.utc_now().tzinfo)


class TestPackageVersion(unittest.TestCase):
    """Looking up installed distributions."""

    def test_reports_an_installed_version(self):
        """numpy is a hard runtime dependency."""
        self.assertIsNotNone(provenance._package_version("numpy"))

    def test_returns_none_when_not_installed(self):
        """An optional dependency's absence is not an error."""
        self.assertIsNone(
            provenance._package_version("not-a-real-distribution-xyz")
        )


class TestGitCommit(unittest.TestCase):
    """Best-effort source revision."""

    def test_returns_the_commit_hash(self):
        """A checkout can say which revision produced a flatfield."""
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="abc123\n", stderr=""
        )
        with patch.object(
            provenance.subprocess, "run", return_value=completed
        ):
            self.assertEqual(provenance._git_commit(), "abc123")

    def test_returns_none_outside_a_checkout(self):
        """An installed wheel has no history, and must still run."""
        completed = subprocess.CompletedProcess(
            args=[], returncode=128, stdout="", stderr="not a git repo"
        )
        with patch.object(
            provenance.subprocess, "run", return_value=completed
        ):
            self.assertIsNone(provenance._git_commit())

    def test_returns_none_when_git_is_missing(self):
        """No git binary in the container is not a failure either."""
        with patch.object(
            provenance.subprocess, "run", side_effect=FileNotFoundError
        ):
            self.assertIsNone(provenance._git_commit())

    def test_returns_none_on_empty_output(self):
        """A bare repository reports success with nothing to say."""
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n", stderr=""
        )
        with patch.object(
            provenance.subprocess, "run", return_value=completed
        ):
            self.assertIsNone(provenance._git_commit())


class TestSha256File(unittest.TestCase):
    """Hashing an input so a change is detectable."""

    def test_hashes_a_file(self):
        """Same contents, same digest."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dark.npy"
            path.write_bytes(b"darkfield")
            twin = Path(tmp) / "twin.npy"
            twin.write_bytes(b"darkfield")
            self.assertEqual(
                provenance.sha256_file(path), provenance.sha256_file(twin)
            )

    def test_different_contents_differ(self):
        """A silently swapped darkfield has to be visible."""
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.npy"
            first.write_bytes(b"one")
            second = Path(tmp) / "b.npy"
            second.write_bytes(b"two")
            self.assertNotEqual(
                provenance.sha256_file(first),
                provenance.sha256_file(second),
            )

    def test_returns_none_for_an_unreadable_file(self):
        """Provenance is best effort; it must not fail the run."""
        self.assertIsNone(provenance.sha256_file("/no/such/file.npy"))


class TestGatherProvenance(unittest.TestCase):
    """The record written into the sidecar."""

    def _gather(self, **overrides):
        """
        Gather provenance for a fixed 90-second run.

        Parameters
        ----------
        **overrides : Any
            Argument overrides.

        Returns
        -------
        dict
            The provenance record.
        """
        return provenance.gather_provenance(
            _args(**overrides), STARTED, FINISHED
        )

    def test_records_the_invocation_and_timing(self):
        """Enough to re-run the same command."""
        record = self._gather()
        self.assertIsInstance(record["command"], list)
        self.assertEqual(record["duration_seconds"], 90.0)
        self.assertEqual(record["started_utc"], STARTED.isoformat())

    def test_records_every_argument(self):
        """The full namespace, so a new flag needs no code change here."""
        record = self._gather(pyramid_level=2)
        self.assertEqual(record["args"]["pyramid_level"], 2)
        self.assertEqual(record["args"]["base_path"], "/data/ch_405")

    def test_records_the_solver_dependencies(self):
        """A pinned version changing is enough to change the result."""
        deps = self._gather()["dependencies"]
        self.assertIn("BaSiCPy", deps)
        self.assertIn("jax", deps)
        self.assertIsNotNone(deps["numpy"])

    def test_records_the_sampling_seeds(self):
        """The subsamples are reproducible only if the seeds are known."""
        seeds = self._gather()["seeds"]
        self.assertEqual(seeds["stratified_subsample"], 42)
        self.assertEqual(seeds["confirmation_subsample"], 7)

    def test_records_the_compute_environment(self):
        """Worker counts explain a per-Z run's timing."""
        record = self._gather()
        self.assertIn("processes", record["workers"])
        self.assertIn("threads_each", record["workers"])
        self.assertIsNotNone(record["platform"])

    def test_hashes_a_supplied_darkfield_image(self):
        """The flatfield is only valid for that exact darkfield."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dark.npy"
            path.write_bytes(b"darkfield")
            record = self._gather(darkfield_image=str(path))
        self.assertEqual(len(record["darkfield_image_sha256"]), 64)

    def test_omits_the_hash_without_a_darkfield_image(self):
        """A scalar pedestal is already fully recorded in args."""
        self.assertNotIn("darkfield_image_sha256", self._gather())

    def test_survives_an_uninstalled_package(self):
        """Running from a source tree, not an installed distribution."""
        with patch.object(
            provenance,
            "version",
            side_effect=PackageNotFoundError("nope"),
        ):
            record = self._gather()
        self.assertIsNone(record["package_version"])


class TestRunLog(unittest.TestCase):
    """Capturing the run's log to a file."""

    def test_captures_records_from_any_logger(self):
        """Third-party warnings are what explain a non-obvious failure."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimation.log"
            handler = provenance.attach_run_log(path)
            try:
                logging.disable(logging.NOTSET)
                logging.getLogger("jax").warning("third party warning")
                logging.getLogger("aind_flatfield_correction.core").info(
                    "ours"
                )
            finally:
                logging.disable(logging.CRITICAL)
                provenance.detach_run_log(handler)
            written = path.read_text()
        self.assertIn("third party warning", written)
        self.assertIn("ours", written)

    def test_detaching_removes_the_handler(self):
        """Repeated runs in one process must not stack handlers."""
        with tempfile.TemporaryDirectory() as tmp:
            handler = provenance.attach_run_log(Path(tmp) / "estimation.log")
            self.assertIn(handler, logging.getLogger().handlers)
            provenance.detach_run_log(handler)
            self.assertNotIn(handler, logging.getLogger().handlers)

    def test_leaves_the_logging_level_alone(self):
        """The caller's configuration is inherited, never imposed."""
        root = logging.getLogger()
        original = root.level
        try:
            root.setLevel(logging.WARNING)
            with tempfile.TemporaryDirectory() as tmp:
                handler = provenance.attach_run_log(
                    Path(tmp) / "estimation.log"
                )
                self.assertEqual(root.level, logging.WARNING)
                provenance.detach_run_log(handler)
        finally:
            root.setLevel(original)

    def test_inherits_the_configured_formatter(self):
        """The file reads the same as the console it accompanies."""
        root = logging.getLogger()
        saved = list(root.handlers)
        for existing in saved:
            root.removeHandler(existing)
        configured = logging.StreamHandler()
        configured.setFormatter(logging.Formatter("CUSTOM %(message)s"))
        root.addHandler(configured)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "estimation.log"
                handler = provenance.attach_run_log(path)
                try:
                    logging.disable(logging.NOTSET)
                    logging.getLogger("jax").warning("inherited")
                finally:
                    logging.disable(logging.CRITICAL)
                    provenance.detach_run_log(handler)
                self.assertEqual(path.read_text().strip(), "CUSTOM inherited")
        finally:
            root.removeHandler(configured)
            for existing in saved:
                root.addHandler(existing)

    def test_falls_back_to_its_own_format(self):
        """Nothing has configured logging yet."""
        root = logging.getLogger()
        saved = list(root.handlers)
        for existing in saved:
            root.removeHandler(existing)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                handler = provenance.attach_run_log(
                    Path(tmp) / "estimation.log"
                )
                self.assertEqual(handler.formatter._fmt, provenance.LOG_FORMAT)
                provenance.detach_run_log(handler)
        finally:
            for existing in saved:
                root.addHandler(existing)


if __name__ == "__main__":
    unittest.main()
