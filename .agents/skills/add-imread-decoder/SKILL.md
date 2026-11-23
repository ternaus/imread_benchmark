---
name: add-imread-decoder
description: Adds and verifies a JPEG decoder adapter in the schema-2 imread_benchmark registry, environments, support audits, and real worker protocols. Use when adding, porting, or fixing a decoder adapter in imread_benchmark.
---

# Add an imread decoder

Read `CONTRIBUTING.md`, `imread_benchmark/decoders/__init__.py`, and one similar
existing adapter before editing. Use TDD. Do not add a separate runner,
requirements file, skip list, or result format.

## Adapter

Create `imread_benchmark/decoders/<name>_decoder.py` with a `BaseDecoder`
subclass. `decode(data: bytes)` must return a fully materialized, owned,
C-contiguous `(H, W, 3)` RGB `uint8` NumPy array. It must remain valid after the
input and library-local objects are released. Lazy handles, borrowed buffers,
BGR, CHW, and deferred conversion violate the contract.

For Pillow-like lazy APIs, load inside a context manager, convert to RGB, then
copy into an owned NumPy array. Override thread methods only when the backend
really exposes verifiable control.

Declare genuine capabilities on the class:

```python
class FooDecoder(BaseDecoder):
    name = "foo"
    package_name = "foo-package"
    group = "mainstream"
    skip_single = ()
    skip_dataloader = ()
    in_dataloader = True
```

Use `tensorflow` only for the separately resolved TensorFlow stack; compatible
decoders belong to `mainstream`.

## Registration and lock

In `pyproject.toml`, add one dotted key under `[project]`:
`entry-points."imread_benchmark.decoders".<name> = "imread_benchmark.decoders.<name>_decoder:<Class>"`,
and add the distribution to the matching `optional-dependencies.<group>` list. Refresh `uv.lock`
and verify `uv sync --frozen --no-editable` can provision that group. Registry
discovery must continue through entry points; do not add a manual registry.

## Required tests

1. bytes and path decode produce owned/materialized normalized output;
2. a red-channel fixture catches BGR output;
3. pixel agreement with Pillow within a documented tolerance;
4. default/requested thread behavior where applicable;
5. a real spawned DataLoader worker smoke when `in_dataloader` is true;
6. capability metadata, dependency-group discovery, and platform skips;
7. a support audit records item-level failures before timing.

Run the relevant decoder tests first, then the full gates:

```bash
uv run pytest -q -k foo
uv run pytest -q
uv run pre-commit run --all-files
```

Use a small schema-2 campaign smoke for runtime verification. Do not invoke an
adapter directly as a benchmark and do not manufacture a standalone JSON file.
