"""Tests for the tuning constants and compute-environment settings."""

import os
import unittest
from unittest.mock import mock_open, patch

from aind_flatfield_correction.core.basicpy import config

CGROUP_ENV = {"CO_CPUS": "", "AWS_BATCH_JOB_ID": ""}


class TestConstants(unittest.TestCase):
    """The pinned BaSiC configuration and search grid."""

    def test_darkfield_is_off(self):
        """The darkfield knobs are no-ops only while this stays False."""
        self.assertFalse(config.BASIC_CONFIG["get_darkfield"])

    def test_ladmap_solver_and_iterations(self):
        """ladmap produced the known-good flats, not approximate."""
        self.assertEqual(config.BASIC_CONFIG["fitting_mode"], "ladmap")
        self.assertEqual(config.BASIC_CONFIG["max_reweight_iterations"], 35)

    def test_search_grid_contains_the_baseline(self):
        """The incumbent must be scored alongside the candidates."""
        self.assertIn(
            config.MANUAL_PARAMS["smoothness_flatfield"],
            config.SEARCH_GRID,
        )

    def test_search_grid_is_sorted_and_unique(self):
        """Ties are broken deterministically, so order matters."""
        self.assertEqual(config.SEARCH_GRID, sorted(config.SEARCH_GRID))
        self.assertEqual(len(config.SEARCH_GRID), len(set(config.SEARCH_GRID)))

    def test_plausibility_thresholds_are_ordered(self):
        """A flatfield cannot be required to be both too flat and too wide."""
        self.assertLess(config.FF_STD_FLOOR, config.FF_STD_CEILING)
        self.assertGreater(config.FF_SPAN_FLOOR, 0.0)
        self.assertLess(config.FF_REL_FLOOR, 1.0)


class TestLimitWorkerThreads(unittest.TestCase):
    """Thread caps applied inside a worker process."""

    def test_sets_cpu_platform_and_thread_caps(self):
        """JAX stays on CPU and BLAS is capped."""
        with patch.dict(os.environ, {}, clear=True):
            config.limit_worker_threads(3)
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "3")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "3")

    def test_never_caps_below_one_thread(self):
        """A zero or negative request would disable threading."""
        with patch.dict(os.environ, {}, clear=True):
            config.limit_worker_threads(0)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")

    def test_does_not_override_an_existing_setting(self):
        """setdefault semantics: the environment wins."""
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "12"}, clear=True):
            config.limit_worker_threads(2)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "12")


class TestGetCpuLimit(unittest.TestCase):
    """CPU discovery across Code Ocean, Batch, cgroups and bare metal."""

    def test_code_ocean_limit_is_returned_as_an_int(self):
        """CO_CPUS is a string; worker_pool_size compares it to an int."""
        with patch.dict(os.environ, {"CO_CPUS": "16"}, clear=True):
            limit = config.get_cpu_limit()
        self.assertEqual(limit, 16)
        self.assertIsInstance(limit, int)

    def test_aws_batch_job_gets_one_cpu(self):
        """Batch jobs are pinned to a single core."""
        with patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "job-1"}, clear=True):
            self.assertEqual(config.get_cpu_limit(), 1)

    def test_reads_the_cgroup_quota(self):
        """A container quota of 200000/100000 is two cores."""
        reads = mock_open(read_data="200000")
        with patch.dict(os.environ, CGROUP_ENV, clear=True):
            with patch("builtins.open", reads):
                reads.return_value.read.side_effect = [
                    "200000",
                    "100000",
                ]
                self.assertEqual(config.get_cpu_limit(), 2)

    def test_falls_back_to_physical_cores_without_cgroups(self):
        """No cgroup files means a physical machine."""
        with patch.dict(os.environ, CGROUP_ENV, clear=True):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch.object(config.psutil, "cpu_count", return_value=6):
                    self.assertEqual(config.get_cpu_limit(), 6)

    def test_falls_back_when_the_quota_is_unset(self):
        """A quota of -1 means unlimited, so use the physical count."""
        reads = mock_open()
        with patch.dict(os.environ, CGROUP_ENV, clear=True):
            with patch("builtins.open", reads):
                reads.return_value.read.side_effect = [
                    "-1",
                    "100000",
                ]
                with patch.object(config.psutil, "cpu_count", return_value=4):
                    self.assertEqual(config.get_cpu_limit(), 4)


class TestWorkerPoolSize(unittest.TestCase):
    """Worker and thread budget derived from the CPU limit."""

    def test_caps_workers_and_spends_the_rest_on_threads(self):
        """16 cores, default ceiling of 8, leaves two threads each."""
        with patch.object(config, "get_cpu_limit", return_value=16):
            self.assertEqual(config.worker_pool_size(), (8, 2))

    def test_honours_a_lower_ceiling(self):
        """An explicit ceiling gives each worker more threads."""
        with patch.object(config, "get_cpu_limit", return_value=16):
            self.assertEqual(config.worker_pool_size(2), (2, 8))

    def test_never_asks_for_more_workers_than_cores(self):
        """Two cores cannot run eight workers."""
        with patch.object(config, "get_cpu_limit", return_value=2):
            self.assertEqual(config.worker_pool_size(), (2, 1))

    def test_survives_an_unknown_core_count(self):
        """psutil returns None on some platforms."""
        with patch.object(config, "get_cpu_limit", return_value=None):
            self.assertEqual(config.worker_pool_size(), (1, 1))

    def test_survives_a_none_ceiling(self):
        """An explicit None must not become min(n_cpu, None)."""
        with patch.object(config, "get_cpu_limit", return_value=8):
            self.assertEqual(config.worker_pool_size(None), (1, 8))


if __name__ == "__main__":
    unittest.main()
