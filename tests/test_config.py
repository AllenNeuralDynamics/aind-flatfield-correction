"""Tests for the tuning constants and compute-environment settings."""

import json
import os
import tempfile
import unittest
from pathlib import Path
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

    def test_carries_the_searched_parameter(self):
        """The smoothness is a solver setting like any other."""
        self.assertEqual(config.BASIC_CONFIG[config.SMOOTHNESS_KEY], 1.0)

    def test_plausibility_thresholds_are_ordered(self):
        """A flatfield cannot be required to be both too flat and too wide."""
        self.assertLess(config.FF_STD_FLOOR, config.FF_STD_CEILING)
        self.assertGreater(config.FF_SPAN_FLOOR, 0.0)
        self.assertLess(config.FF_REL_FLOOR, 1.0)


class TestBaselineParams(unittest.TestCase):
    """The incumbent parameters a configuration carries."""

    def test_reads_the_configured_smoothness(self):
        """A supplied value is what the search must beat."""
        resolved = config.load_basic_config('{"smoothness_flatfield": 2.5}')
        self.assertEqual(
            config.baseline_params(resolved),
            {"smoothness_flatfield": 2.5},
        )

    def test_defaults_to_the_known_good_value(self):
        """The built-in configuration is the usual incumbent."""
        self.assertEqual(
            config.baseline_params(config.BASIC_CONFIG),
            {"smoothness_flatfield": 1.0},
        )

    def test_survives_a_configuration_without_the_key(self):
        """A hand-built config dict need not carry every key."""
        self.assertEqual(
            config.baseline_params({"fitting_mode": "ladmap"}),
            {"smoothness_flatfield": 1.0},
        )

    def test_coerces_the_value_to_a_float(self):
        """JSON gives an int for 2, and the grid is float-keyed."""
        params = config.baseline_params({"smoothness_flatfield": 2})
        self.assertIsInstance(params["smoothness_flatfield"], float)


class TestSearchGrid(unittest.TestCase):
    """The 1-D candidate grid, built per run."""

    def test_includes_the_incumbent(self):
        """Its score has to be comparable with the candidates'."""
        self.assertIn(2.5, config.search_grid(2.5))

    def test_covers_three_decades(self):
        """The logspace points are the same regardless of incumbent."""
        grid = config.search_grid(1.0)
        self.assertEqual(min(grid), 0.01)
        self.assertEqual(max(grid), 10.0)

    def test_is_sorted_and_unique(self):
        """Ties are broken deterministically, so order matters."""
        grid = config.search_grid(2.5)
        self.assertEqual(grid, sorted(grid))
        self.assertEqual(len(grid), len(set(grid)))

    def test_an_incumbent_on_the_grid_adds_no_candidate(self):
        """1.0 is already a logspace point."""
        self.assertEqual(
            len(config.search_grid(1.0)), len(config.search_grid(10.0))
        )

    def test_an_incumbent_off_the_grid_adds_one_candidate(self):
        """A hand-tuned value becomes an extra point."""
        self.assertEqual(
            len(config.search_grid(2.5)),
            len(config.search_grid(1.0)) + 1,
        )


class TestLoadBasicConfig(unittest.TestCase):
    """Resolving the BaSiC solver configuration for a run."""

    def test_defaults_to_the_known_good_configuration(self):
        """No input means the vetted baseline."""
        self.assertEqual(config.load_basic_config(), config.BASIC_CONFIG)

    def test_returns_a_copy_of_the_defaults(self):
        """A run must not be able to mutate the module constant."""
        resolved = config.load_basic_config()
        resolved["fitting_mode"] = "mutated"
        self.assertEqual(config.BASIC_CONFIG["fitting_mode"], "ladmap")

    def test_an_empty_source_keeps_the_defaults(self):
        """An unset argument arrives as None or an empty string."""
        self.assertEqual(config.load_basic_config(""), config.BASIC_CONFIG)

    def test_merges_inline_json_over_the_defaults(self):
        """Only the changed keys have to be supplied."""
        resolved = config.load_basic_config('{"fitting_mode": "approximate"}')
        self.assertEqual(resolved["fitting_mode"], "approximate")
        self.assertEqual(
            resolved["max_reweight_iterations"],
            config.BASIC_CONFIG["max_reweight_iterations"],
        )

    def test_tolerates_surrounding_whitespace(self):
        """Shell quoting leaves it behind."""
        resolved = config.load_basic_config('  {"sort_intensity": false} ')
        self.assertFalse(resolved["sort_intensity"])

    def test_merges_a_json_file(self):
        """Code Ocean runs pass configuration as a file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basic.json"
            path.write_text(json.dumps({"get_darkfield": True}))
            resolved = config.load_basic_config(str(path))
        self.assertTrue(resolved["get_darkfield"])

    def test_can_add_a_key_the_defaults_do_not_set(self):
        """basicpy has more knobs than the baseline pins."""
        resolved = config.load_basic_config('{"max_iterations": 42}')
        self.assertEqual(resolved["max_iterations"], 42)

    def test_keeps_a_supplied_smoothness(self):
        """It is the searched parameter, supplied like any other."""
        resolved = config.load_basic_config(
            '{"smoothness_flatfield": 7.0, "sort_intensity": false}'
        )
        self.assertEqual(resolved["smoothness_flatfield"], 7.0)
        self.assertFalse(resolved["sort_intensity"])

    def test_rejects_a_missing_file(self):
        """A typo'd path must not silently fall back to the defaults."""
        with self.assertRaises(FileNotFoundError):
            config.load_basic_config("/no/such/basic.json")

    def test_rejects_json_that_is_not_an_object(self):
        """A list cannot be splatted into BaSiC."""
        with self.assertRaises(ValueError):
            config.load_basic_config('["ladmap"]')

    def test_rejects_unparseable_json(self):
        """Fail on the malformed input, not later inside the solver."""
        with self.assertRaises(json.JSONDecodeError):
            config.load_basic_config('{"fitting_mode": ')


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
