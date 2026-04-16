# Contributing

## Development setup

```bash
pip install uv
uv sync --extra dev
```

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

### 3. Add requirements file

Create `requirements/foo.txt`:

```
foo-package
```

### 4. Add to run_benchmarks.sh

Add `"foo"` to `ALL_LIBRARIES` in [`run_benchmarks.sh`](run_benchmarks.sh). If the library is macOS-incompatible, add a filter in `get_libraries()`.

### 5. Document system deps

If a system library is required (e.g. `brew install something`), add it to the **System Requirements** section in [`README.md`](README.md).

### 6. Verify

```bash
# Tests auto-discover the decoder once it is in REGISTRY and the package is installed
uv run pytest tests/ -v

# Quick smoke run
BENCHMARK_LIBRARY=foo python imread_benchmark/benchmark_single.py \
    --data-dir /path/to/imagenet/val \
    --output-dir output \
    --num-images 100 \
    --num-runs 2
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
requirements/
├── base.txt                 # shared benchmark deps
└── <library>.txt            # per-library isolated venv deps
```
