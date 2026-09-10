# aind-flatfield-correction

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Flatfield estimation for microscopy tile datasets, built on
[BaSiCPy](https://github.com/peng-lab/BaSiCPy).

One run fits a single multiplicative flatfield for one channel from a bounded number of Z
planes taken from **every** tile in that channel. Every tile contributes: N tiles at one Z
index are N different regions seen through the same optics, and so are highly informative
about the `(y, x)` profile, whereas consecutive Z planes of a single tile are nearly
redundant. The image budget is therefore spent on tiles, not on Z.

## Flatfield Examples
We estimate the flatfields by smartly selecting slices from each tile in the dataset.
![Flatfield and raw data](https://github.com/AllenNeuralDynamics/aind-flatfield-correction/blob/main/imgs/flatfield_estimation_raw_data.png?raw=true)

This is the result after applying flatfield to the tiles of a proteomics dataset.
![Example Flatfield](https://github.com/AllenNeuralDynamics/aind-flatfield-correction/blob/main/imgs/flatfield_corrected.png?raw=true)
![Example Flatfield Distribution](https://github.com/AllenNeuralDynamics/aind-flatfield-correction/blob/main/imgs/plot_distribution.png?raw=true)

In very low illumination datasets and lower wavelenght channels, we can even detect the difference in the intensity gains of the camera chips.
![Two Difference Gain](https://github.com/AllenNeuralDynamics/aind-flatfield-correction/blob/main/imgs/two_gain_difference.png?raw=true)

> **Note:** Technically speaking, we don't want to use the same darkfield for each side of the intensity gain, but this is good enough as an approximation and scientists as happy with the results.

## Installation

```bash
uv sync
```

> **Note:** `aind-large-scale-prediction`, which provides the OME-Zarr reader, is not
> declared in `pyproject.toml` — its published release pins a yanked `imagecodecs`
> version that `uv` cannot resolve. Install it separately (from a GitHub release, or via
> the Code Ocean environment) before running the estimator.

For development, `uv sync` already includes the dev group. Without `uv`:

```bash
pip install -e . --group dev   # --group needs pip >= 25.1
```

## Quick start

`base_path` is **one channel's folder**: every OME-Zarr directly inside it is treated as a
tile of the same channel. Nothing is inferred from tile names, so to estimate several
channels you run the command once per channel folder.

```bash
python -m aind_flatfield_correction.core.basicpy.estimate \
    s3://aind-open-data/HCR_800792_2026-03-25_13-00-00/SPIM/ch_405 \
    --pyramid-level 3 \
    --darkfield-value 90 \
    --validate \
    --output-folder ./flatfields
```

If the tiles for several channels sit in one flat folder, select a subset with a regular
expression and name the run yourself:

```bash
python -m aind_flatfield_correction.core.basicpy.estimate <base_path> \
    --tile-pattern '_ch_405\.ome\.zarr$' \
    --output-name ch405
```

Local paths work the same way as `s3://` ones.

## What it produces

Named after `--output-name`, or after the last segment of `base_path` if you omit it:

| Path | Contents |
|---|---|
| `flatfield_<name>.npy` | The fitted flatfield, at the estimation pyramid level |
| `darkfield_<name>.npy` | The darkfield plane that was actually subtracted before fitting |
| `flatfield_<name>.json` | Sidecar: solver config, chosen parameters, search report, flatfield statistics, the plausibility verdict, and the list of tiles used |
| `<name>/*.png` | Inspection figures, only with `--validate` |

`darkfield_<name>.npy` holds the pedestal used for the fit rather than BaSiC's own
darkfield estimate — `get_darkfield` is off, so that estimate is all zeros, and the
pedestal is what an apply step needs.

## How it works

1. **Gather planes.** Every tile is probed for its shape, then `--max-fit-planes ÷ n_tiles`
   planes are streamed from each, taken from the middle 60% of Z. The stack ends
   zero-padded to the largest plane extent.
2. **Remove the sensor pedestal.** `raw = pedestal + signal × flatfield`. The offset is
   added after light collection, so it is not vignetted, and a multiplicative field must
   be estimated on the signal alone. Fitting on raw values squashes the apparent
   vignetting by `S / (P + S)` — 14× on a channel whose median raw value is 97 over a
   pedestal of 90 — and yields a near-unity, no-op flatfield.
3. **Choose `smoothness_flatfield`.** A two-stage search: a cheap screen over a
   logarithmic grid at reduced solver iterations, then the best few re-scored at full
   fidelity. The objective is the differential entropy of the corrected intensity
   histogram and it is **minimized** — basicpy's autotune returns `-entropy` to a
   maximizing optimizer. A calibration gate projects the cost first and abandons the
   search if it would exceed `--max-search-minutes`.
4. **Confirm at scale.** In ladmap mode `init_mu` depends on N, so a winner found on ~150
   slices is refitted against the incumbent on `--n-confirm` slices and reverted unless it
   still leads.
5. **Fit, then guard.** The accepted flatfield is checked for plausibility — non-finite
   values, a non-positive minimum, and floors and a ceiling on its standard deviation and
   span. basicpy normalizes the flatfield mean to 1.0, so those thresholds are scale-free.
   A field that fails is refitted with the configured baseline; if it still fails, the run
   records it as `"suspicious": true` rather than passing it off as good.

The incumbent the search must beat is whatever `smoothness_flatfield` the solver
configuration carries, so a candidate can never make the result worse than the
configuration you supplied.

## Options

| Flag | Default | Purpose |
|---|---|---|
| `base_path` | — | Channel folder holding the tiles; accepts `s3://` |
| `--tile-pattern` | all entries | Regex selecting which tiles inside the folder to fit |
| `--output-name` | last path segment | Names the outputs and figure titles |
| `--pyramid-level` | `3` | Multiscale level the estimation runs on |
| `--darkfield-value` | `0.0` | Scalar pedestal in ADU counts |
| `--darkfield-image` | — | `.npy`/`.tif` darkfield; takes precedence, resized to the estimation level, and a stack of dark frames is averaged to one plane |
| `--output-folder` | `flatfield_estimation` | Where the products are written |
| `--method` | `fit` | `fit` is one joint fit; `per-z-median` fits each Z index across all tiles and takes the pixelwise median |
| `--basic-config` | built-in | Solver configuration; see below |
| `--max-fit-planes` | `2000` | Image budget across all tiles |
| `--validate` | off | Write the inspection figures |
| `--validate-tiles` | `4` | How many tiles get a profile figure |
| `--skip-search` | off | Fit the configured smoothness directly |
| `--max-eval-slices` | `150` | Slices used to score each candidate |
| `--n-confirm` | `1500` | Slices for the confirmation refit; `0` disables |
| `--max-search-minutes` | `120.0` | Abandon the search if projected to exceed this |

### Solver configuration

`--basic-config` takes a path to a JSON file or an inline JSON object, and is **merged
over** the built-in defaults — only the keys you want to change need to be given:

```bash
--basic-config '{"fitting_mode": "approximate"}'
--basic-config ./basic_config.json
```

The defaults are the vetted values: `ladmap` (which produced the known-good flatfields,
where `approximate` was a regression), `max_reweight_iterations` raised to 35 from
basicpy's 10, and `get_darkfield` off because the pedestal is removed beforehand.

`smoothness_flatfield` is supplied the same way. It is the parameter the search varies, so
the configured value is the incumbent every candidate is measured against, the value a
failed fit falls back to, and — with `--skip-search` — the value fitted directly:

```bash
--basic-config '{"smoothness_flatfield": 2.5}' --skip-search
```

The configuration is validated against BaSiC before any tile is read, so a mistyped key
fails immediately instead of after a long load.

## Parallel Grid Search
We parallelized the grid search by using a shared memory compartment and maximizing the usage of the CPU cores with JAX. This is necessary as some parameters might be hard to estimate for every dataset, experiments and sequential exploration of parameters is extremely slow.

Here we show an example of using this package with another dataset, specifically a hybridization chain reaction dataset.

![HCR dataset](https://github.com/AllenNeuralDynamics/aind-flatfield-correction/blob/main/imgs/hcr_dataset.png?raw=true)

## Reviewing a flatfield

`--validate` writes three kinds of figure, none of which need tile positions on the
acquisition grid:

- `<name>_overview.png` — the flatfield beside one example plane, raw and corrected.
- `<name>_vignetting_profiles.png` — X and Y intensity profiles of the mean of every
  fitted plane, before and after correction, with the coefficient of variation of each.
  A correct flatfield lowers it.
- `<name>_tile<NNN>_<tile>.png` — per-tile panels for `--validate-tiles` tiles spread
  across the list: raw, X profile, Y profile, corrected. The raw curve should be bowed by
  vignetting and the corrected one flat. **This is the acceptance gate** — review these
  before applying a flatfield to a whole dataset.

Profiles average occupied pixels only; tiles smaller than the padded plane leave zeros
that would otherwise flatten the curves.

## Python API

```python
from pathlib import Path

from aind_flatfield_correction.core.basicpy.config import load_basic_config
from aind_flatfield_correction.core.basicpy.tiles import (
    list_tiles, load_darkfield, load_fit_stack, match_darkfield, subtract_pedestal,
)
from aind_flatfield_correction.core.basicpy.fit import estimate_joint, report_flatfield

basic_config = load_basic_config('{"smoothness_flatfield": 1.0}')
tiles = list_tiles("s3://bucket/dataset/SPIM/ch_405")

stack = load_fit_stack("s3://bucket/dataset/SPIM/ch_405", tiles, 3, 2000)
dark = match_darkfield(load_darkfield(None, 90), (stack.height, stack.width))
fit_slices, _ = subtract_pedestal(stack.slices, dark)

flatfield, _, _ = estimate_joint(fit_slices, basic_config)
ok, stats, reasons = report_flatfield(flatfield)
```

### Module layout

| Module | Responsibility |
|---|---|
| `core.basicpy.estimate` | CLI and per-run orchestration |
| `core.basicpy.tiles` | Tile discovery, plane loading, darkfield handling |
| `core.basicpy.search` | The entropy objective and the parameter search |
| `core.basicpy.fit` | Joint and per-Z fitting, plausibility guard |
| `core.basicpy.figures` | Inspection figures |
| `core.basicpy.config` | Solver defaults, thresholds, compute-environment settings |
| `metrics.metrics` | Coefficient of variation and masked profiles |

## Development

```bash
uv run coverage run -m unittest discover && uv run coverage report  # 100% required
uv run interrogate .    # docstring coverage, 100% required
uv run flake8 .         # style and complexity
uv run black . && uv run isort .
```

Tests are one file per module under `tests/`, with shared doubles in `tests/fakes.py`.
Every third-party seam — the OME-Zarr reader, `BaSiC`, the process pools — is patched per
test, so the suite needs no data and no GPU.

To rebuild the Sphinx documentation:

```bash
uv run sphinx-apidoc -o docs/source/ src
uv run sphinx-build -b html docs/source/ docs/build/html
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for linting, testing and the Angular-style commit
convention used to drive semantic versioning.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.
