# Contributing

## Development setup

```bash
uv sync --frozen --group dev --extra mainstream
uv run pytest -q
uv run pre-commit run --all-files
```

Do not add an alternate runner, result JSON format, or plotting reader. The
schema-2 package, campaign, bundle, canonical loader, and publication path are
the only supported workflow.

## Adding a decoder

Create `imread_benchmark/decoders/<name>_decoder.py`:

```python
from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class FooDecoder(BaseDecoder):
    name = "foo"
    package_name = "foo-package"
    group = "mainstream"

    def decode(self, data: bytes) -> np.ndarray:
        import foo

        result = foo.decode(data)
        return np.ascontiguousarray(result, dtype=np.uint8)
```

The adapter contract is a fully materialized, C-contiguous `(H, W, 3)` RGB
`uint8` array. The output must remain valid after the input buffer and every
library-local object are released. Lazy image handles, BGR output, CHW tensors,
borrowed buffers, and deferred conversion are invalid.

Pillow is the reference lifetime pattern: open in a context manager, call
`load()`, convert to RGB, materialize again, and copy to an owned NumPy array.

Register the class under
`[project.entry-points."imread_benchmark.decoders"]` and add its distribution
to the appropriate optional dependency group:

- `mainstream` for packages compatible with the PyTorch-oriented environment;
- `tensorflow` only for the separately resolved TensorFlow stack.

Use the decoder class capability fields for genuine platform limitations:

```python
class FooDecoder(BaseDecoder):
    skip_single = (("Darwin", "*"),)
    skip_dataloader = (("Linux", "aarch64"),)
    in_dataloader = True
```

Do not silently skip a failure discovered during a campaign. The plan preflight
must either reject an unsupported configuration or the support audit must record
the item-level failure before timing.

## Required tests for a decoder

1. bytes and path decode return normalized, owned/materialized output;
2. red-channel order test catches BGR output;
3. pixel agreement with the Pillow reference within the documented tolerance;
4. fixed/default thread-control behavior where applicable;
5. real DataLoader worker smoke if `in_dataloader` is true;
6. platform capability metadata and dependency-group discovery.

Run the full suite with the dependency group installed. A unit test that mocks
the decoder import is not a replacement for the real subprocess/worker smoke.

## Architecture invariants

- One timed configuration runs in one fresh subprocess.
- Data transport, verification, validation, and warmup remain outside timing.
- DataLoader process start method is explicit and part of `config_id`.
- Support sets are immutable and pinned by ordered item IDs.
- Completed bundles are write-once and `COMMITTED.json` is written last.
- Environment provisioning uses `uv sync --frozen --no-editable`.
- GCP scripts are lifecycle/bootstrap only; experiment semantics belong in
  typed Python modules.
- Publication code reads only the canonical bundle loader.

Any change to an identity field needs mutation tests showing that the relevant
ID changes. Any change to remote storage needs corruption, conflict, incomplete
commit, and resume tests.

## Documentation and claims

Reader-facing benchmark claims must state protocol, workload, platform, output
contract, support policy, repetition count, and uncertainty. Decoder or loader
measurements cannot be described as training speed.

JPEG quality reconstructed from quantization tables is an estimate. Natural
FODB processing variants confound resize, encoder, quantization, subsampling,
and metadata changes; causal quality/resolution claims require the controlled
ablation built by `dataset controlled-package` and documented in
[docs/controlled_ablation.md](docs/controlled_ablation.md).

## Local-only material

`_internal/` is ignored and may hold manuscript drafts or scratch outputs. It
must not become a second implementation, result format, or publication source.
Reusable plans, tests, schemas, and documentation belong in tracked package or
`docs/` files.
