# NeurIPS Evaluations & Datasets Artifact

This repository contains the anonymous review artifact for the paper
"Need for Speed: A Comprehensive Benchmark of JPEG Decoders in Python".

## Contents

- `imread_benchmark/`: benchmark package and decoder adapters.
- `tests/`: unit and smoke tests for benchmark behavior and decoder agreement.
- `output/`: JSON benchmark outputs used for the paper tables and plots.
- `docs/assets/benchmarks/`: rendered benchmark figures.
- `tools/`: utilities for plotting, README table rendering, and paper assets.
- `gcp/`: optional cloud orchestration scripts used for full benchmark runs.

## Environment

The artifact requires Python 3.11 or newer and `uv`. Install only the
lightweight benchmark orchestrator into the active environment:

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

The `imread-benchmark run` command creates decoder worker environments
automatically under `venvs/<group>/` as needed. The current groups are
`mainstream` and `tensorflow`; they are separate because TensorFlow and
PyTorch-oriented packages can have incompatible NumPy/protobuf constraints.
Do not create those environments manually for normal artifact use.

For the broad decoder test matrix on Linux, install system libjpeg-turbo
development packages before running the benchmark. The Python packages are
still installed by the CLI into worker environments:

```bash
sudo apt-get update
sudo apt-get install -y libjpeg-turbo8-dev libturbojpeg0-dev
```

Plotting is the one optional Python extra used by the control-plane environment:

```bash
uv pip install -e '.[plot]'
```

## Quick Checks

Run these commands to verify the artifact without a large dataset:

```bash
uv run pytest tests/ -v
uv run imread-benchmark list-libs
uv pip install -e '.[plot]'
uv run imread-benchmark plot --input output --output docs/assets/benchmarks
```

Expected behavior:

- tests pass, with decoder-specific skips if optional packages are unavailable;
- `list-libs` reports available and unavailable decoders;
- plot regeneration overwrites the PNG files under `docs/assets/benchmarks/`.

## Reproducing Benchmark Tables and Figures

The included paper results are stored as JSON files under `output/`. Regenerate
the README tables and plots from those files with:

```bash
uv run imread-benchmark plot --input output --output docs/assets/benchmarks
uv run imread-benchmark render-readme
```

## Running a Smoke Benchmark

Point the benchmark at any directory containing JPEG images:

```bash
uv run imread-benchmark run --data-dir /path/to/jpeg/dir \
    --num-images 2000 --num-runs 5 --dataloader-runs 2 \
    --workers 0,2
```

On first use, this command creates the required `venvs/<group>/` worker
environment(s), installs the corresponding optional dependencies, and then runs
the selected decoders. Later runs reuse those environments. Pass `--skip-setup`
only if the worker environments are already present.

The full paper runs used 50,000 ImageNet validation JPEGs and five repeated
runs per platform. ImageNet must be obtained separately under its own access
terms.

## Optional Cloud Runs

The `gcp/` scripts can launch full benchmark runs on Google Cloud VMs. They are
not required for artifact inspection, but document the infrastructure used for
the cross-CPU measurements. See `docs/gcp_benchmarks.md`.

## Anonymization

This artifact branch removes author-identifying names, personal sponsorship
metadata, and public repository-account references from tracked files. Public
citation and license attribution should be restored for the camera-ready public
release.
