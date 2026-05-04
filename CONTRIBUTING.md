# Contributing

## Development setup

```bash
pip install uv
uv sync --group dev
```

## Local-only material (`_internal/`)

The `_internal/` directory is **gitignored**. Keep drafts, papers, regenerated plots, PDFs, and scratch assets there — for example:

- `_internal/papers/paper.md` — arXiv / manuscript drafts (do not add `paper.md` at repo root)
- `_internal/papers/generate_paper_assets.py` — thin wrapper; calls `python -m tools.paper_assets --all` to write `generated/*.md` + `figures/*` from `output/*.json`
- `_internal/plots/` — scratch figures from `tools/create_plots.py` (README plots live in `docs/assets/benchmarks/`)
- `_internal/assets/` — sample images, posters, one-off binaries
- `_internal/notebooks/` — local Jupyter notebooks (the repo ignores `*.ipynb`; do not commit them)

Nothing under `_internal/` is tracked by git.

## Running tests

```bash
uv run pytest tests/ -v
```

## Running linters

```bash
uv run pre-commit run --all-files
```

## Adding a new JPEG decoder

### 1. Create the decoder

Create `imread_benchmark/decoders/<name>_decoder.py`:

```python
from __future__ import annotations

import numpy as np
from imread_benchmark.decoders import BaseDecoder


class FooDecoder(BaseDecoder):
    name = "foo"
    package_name = "foo-package"  # pip distribution name

    def decode(self, data: bytes) -> np.ndarray:
        import foo
        # Must return (H, W, 3) uint8 RGB array
        return foo.decode_jpeg(data)

    # Optional: override if the library has a fast path-based API
    # def decode_path(self, path: str) -> np.ndarray:
    #     import foo
    #     return foo.read_jpeg(path)
```

The default `decode_path()` implementation calls `decode(Path(path).read_bytes())`, which is correct for all libraries.

### 2. Register in the registry

In [`imread_benchmark/decoders/__init__.py`](imread_benchmark/decoders/__init__.py):

```python
from imread_benchmark.decoders.foo_decoder import FooDecoder  # add

REGISTRY = {
    ...
    "foo": FooDecoder,  # add
}
```

### 3. Add to a dependency group

In [`pyproject.toml`](pyproject.toml), add the pip distribution name to one of the `[project.optional-dependencies]` groups:

- `mainstream` — coexists with everything else (opencv, skimage, kornia-rs, torch, etc.). Use this unless you have a real conflict.
- `tensorflow` — only if your library hard-conflicts with torch.

(There used to be a third `pillow-simd` group; dropped 2026-04 — see [`docs/gcp_benchmarks.md`](docs/gcp_benchmarks.md#why-no-pillow-simd) for the reasoning. If you're adding a new Pillow fork that masks vanilla `PIL` in the same venv, add a fresh group rather than reusing this slot.)

```toml
mainstream = [
  "opencv-python-headless",
  ...
  "foo-package",                              # platform-agnostic
  "foo-package; sys_platform == 'linux'",     # Linux only
]
```

Add a platform marker if the wheel doesn't build everywhere.

### 4. Encode platform skips on the class

Nothing else needs editing — `REGISTRY` auto-discovers your decoder and the CLI runs it on every machine that supports it. If your library doesn't run everywhere, set the relevant ClassVars on your decoder:

```python
class FooDecoder(BaseDecoder):
    name = "foo"
    package_name = "foo-package"
    group = "mainstream"                       # which optional-dependencies group provides it
    skip_single = [("Darwin", "*")]            # don't run single-thread benchmark on macOS
    skip_dataloader = [("Linux", "aarch64")]   # don't run inside torch DataLoader on Arm Linux
    in_dataloader = True                       # set False if it never makes sense in a DataLoader
```

### 5. Document system deps

If a system library is required (e.g. `brew install something`), add it to the **System Requirements** section in [`README.md`](README.md).

### 6. Verify

```bash
# Tests auto-discover the decoder once it is in REGISTRY and the package is installed
uv run pytest tests/ -v

# Quick smoke run via the orchestrator
imread-benchmark run --libs foo --mode single \
    --data-dir /path/to/imagenet/val \
    --num-images 100 --num-runs 2
```

## Project structure

```
imread_benchmark/
├── decoders/
│   ├── __init__.py          # BaseDecoder, REGISTRY
│   ├── opencv_decoder.py
│   └── ...                  # one file per library
├── benchmark.py             # timing loop with warmup
├── benchmark_single.py      # CLI: single-library benchmark
├── benchmark_dataloader.py  # CLI: DataLoader throughput benchmark
└── utils.py                 # system info, path helpers
tests/
├── conftest.py              # in-memory JPEG fixture
├── test_decoders.py         # parametrized decode smoke tests
├── test_benchmark.py        # timing loop tests
└── test_utils.py            # utility function tests
tools/
├── analyze_images.py        # dataset statistics
└── create_plots.py          # generate performance charts
pyproject.toml               # 2 dependency groups under [project.optional-dependencies]:
                             #   mainstream / tensorflow
```
