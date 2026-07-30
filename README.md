# imread-benchmark

`imread-benchmark` is a reproducible framework for measuring JPEG decoder and
PyTorch `DataLoader` supply throughput. It is built around immutable dataset
packages, deterministic experiment plans, fresh-process runs, and validated
schema-2 evidence bundles.

The repository no longer ships historical result JSON or a second execution
path. New paper tables and figures must be generated from committed schema-2
bundles through the canonical publication layer.

Preprint: [Single-Thread JPEG Decoder Benchmarks Mis-Evaluate ML Data Loaders](https://arxiv.org/abs/2605.08731).

## What is measured

Two protocols are currently supported:

- `decode-memory`: decode already-resident JPEG bytes into a fully materialized,
  C-contiguous `(H, W, 3)` RGB `uint8` NumPy array;
- `loader-supply`: traverse a real PyTorch `DataLoader` over resident JPEG
  bytes, including worker scheduling, batching, queues, and delivery to the
  consumer process.

Neither protocol measures storage download, archive verification, model
training, GPU transfer, or augmentation. Claims about epoch time require a
separate end-to-end experiment.

Every timed configuration runs in a fresh subprocess. Validation and warmup
are outside the timer. Pillow explicitly materializes the image, converts to
RGB, and copies it into an owned NumPy array before returning.

## Evidence model

A benchmark campaign pins:

- exact JPEG bytes through `package_id`, `manifest_id`, and ordered item IDs;
- a pre-timing operational or common support set;
- decoder threads, DataLoader workers, batch size, prefetch, persistence, and
  multiprocessing start method;
- lock-backed environment and stable platform descriptors;
- source revision, randomized repetition block, and run position.

Each completed run is an immutable directory containing raw samples, phase
events, runtime worker probes, full provenance, derived statistics, payload
hashes, and a final `COMMITTED.json`. A result is invisible until the marker
and every checksum validate.

## Recommended paper workloads

The primary strengthening campaign uses selected scenes from the Forchheim
Image Database (FODB):

- `fodb-native`: original camera JPEGs, providing the large-resolution regime;
- `fodb-mixed`: the same matched scenes and devices after Facebook, Instagram,
  Telegram, Twitter, and WhatsApp processing, providing a realistic mixture of
  resolutions, quantization tables, compression ratios, and metadata.

These are workload comparisons, not a causal estimate of resolution or JPEG
quality. Encoder quality is generally unavailable; `quality_estimate` is only
an estimator derived from quantization tables. See
[Experiment design](docs/experiment_design.md) for the core matrix and the
controlled resolution × quality ablation needed for a causal mechanism claim.
The ablation builder and exact interpretation rules are documented in
[Controlled resolution and JPEG-quality ablation](docs/controlled_ablation.md).

## Setup

Install [`uv`](https://docs.astral.sh/uv/) and sync the locked development
environment:

```bash
uv sync --frozen --group dev --extra mainstream
uv run pytest -q
```

List decoder capability contracts:

```bash
uv run imread-benchmark list-decoders
```

## Prepare FODB once

After downloading the FODB ZIP parts, build the selected native and mixed
workloads. The builder extracts only selected complete scenes, verifies ZIP
CRC values, records JPEG descriptors, hard-links the two local views, and
creates one deduplicated uncompressed tar package.

```bash
uv run imread-benchmark dataset fodb-package \
  --archive ~/data/fodb-part01.zip \
  --archive ~/data/fodb-part02.zip \
  --archive ~/data/fodb-part03.zip \
  --output-root ~/data/fodb-benchmark \
  --scene-count 12 \
  --seed 20260729
```

Upload the returned descriptor to a private GCS prefix:

```bash
uv run imread-benchmark dataset publish \
  ~/data/fodb-benchmark/packages/<package-id>/package.json \
  --store gs://YOUR_BUCKET/imread \
  --prefix datasets
```

The command returns the remote descriptor object key used by local and cloud
materializers. Dataset redistribution rights are not implied; keep the bucket
private and follow FODB's terms.

## Build the controlled resolution-quality package

For the separate causal ablation, start from a pinned lossless PNG source set
and generate every matched factor cell with one command:

```bash
uv run imread-benchmark dataset controlled-package \
  --source-dir /data/pinned-lossless-png \
  --output-root /data/controlled-jpeg \
  --source-name SOURCE_DATASET_NAME \
  --source-release SOURCE_DATASET_RELEASE \
  --source-license SOURCE_DATASET_LICENSE \
  --long-edge 512 --long-edge 1024 --long-edge 2048 \
  --quality 50 --quality 75 --quality 90 --quality 95 \
  --include-native \
  --subsampling 4:2:0 \
  --compressed-byte-limit 2147483648
```

This produces 16 workloads with identical source membership and order. Generate
their plans from
[`examples/controlled-ablation.template.yaml`](examples/controlled-ablation.template.yaml)
with `plan instantiate`; do not interpret FODB's naturally confounded
size/quality mixture as this controlled effect.

## Generate and validate plans before spending money

An experiment plan is schema 2 YAML and must pin the IDs printed in
`package.json`. Example protocol profiles:

```yaml
matrix:
  decoders:
    pillow: {threads: [default]}
    opencv: {threads: [default, 1]}
    simplejpeg: {threads: [default]}
    torchvision: {threads: [default, 1]}
  protocols:
    decode-memory: {}
    loader-supply:
      worker_profiles:
        - workers: [0]
          batch_size: 1
        - workers: [2, 4, 8]
          batch_size: 1
          prefetch_factor: 1
          persistent_workers: true
          multiprocessing_start_method: spawn
execution:
  per_run_subprocess: true
  run_timeout_seconds: 3600
  checkpoint_each_run: true
  maximum_memory_fraction: 0.6
```

Generate one pinned, validated plan for each selected workload. Omit
`--workload` to generate plans for every workload in the package:

```bash
uv run imread-benchmark plan instantiate \
  examples/fodb-experiment.template.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --output-dir plans/fodb \
  --workload fodb-native \
  --workload fodb-mixed
```

The command prints each plan ID and its run count per platform. Inspect or
materialize the deterministic randomized matrix when needed:

```bash
uv run imread-benchmark plan validate plans/fodb/fodb-native.yaml \
  --package-descriptor /data/packages/<package-id>/package.json

uv run imread-benchmark plan expand plans/fodb/fodb-native.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --output expanded-plan.json
```

The complete five-block FODB matrix is available as
[`examples/fodb-experiment.template.yaml`](examples/fodb-experiment.template.yaml).
The generated workload plans are reused unchanged across CPU platforms;
captured platform identity remains part of every run key.

## Run locally

Capture platform provenance and provision the frozen worker environment:

```bash
REVISION=$(git rev-parse HEAD)

uv run imread-benchmark platform capture \
  --output artifacts/platform.json \
  --machine-type local \
  --location local

uv run imread-benchmark environment provision \
  --group mainstream \
  --runner-revision "$REVISION" \
  --project-root . \
  --cache-root .cache/environments
```

Then pass the emitted environment descriptor and Python path to the campaign:

```bash
<environment-python> -m imread_benchmark.cli campaign run experiment.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --environment-descriptor <environment.json> \
  --platform-descriptor artifacts/platform.json \
  --artifact-root artifacts \
  --attempts-root attempts \
  --runner-revision "$REVISION" \
  --worker-python <environment-python>
```

## Run on GCP

The launcher uploads a content-addressed source snapshot and plan, creates an
ephemeral VM, materializes the dataset from one sequential GCS object, restores
or builds a content-addressed frozen environment, and checkpoints every completed bundle. `DONE`
or `FAILED` is written last; the VM then deletes itself unless failure retention
was explicitly requested.

```bash
./gcp/run.sh \
  --plan experiment.yaml \
  --dataset-store gs://YOUR_BUCKET/imread \
  --dataset-descriptor datasets/<package-id>/package.json \
  --results-store gs://YOUR_BUCKET/imread-results \
  --environment-store gs://YOUR_BUCKET/imread-cache \
  --machine-type c3-standard-16 \
  --groups mainstream \
  --no-wait
```

Starting another VM with the same source, plan, package, environment, and
platform pulls valid committed bundles first and launches only missing runs.
See [GCP campaigns](docs/gcp_benchmarks.md).

## Validate and publish

```bash
uv run imread-benchmark artifacts validate artifacts

uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts \
  --output-dir generated

uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts \
  --output-dir generated \
  --check
```

Publication output includes raw sample values, repetition-level configuration
groups, and a provenance sidecar with all bundle IDs, filters, claim scope,
generator revision, and output hash.
Training claims are rejected for decoder-only or loader-only evidence.

## Project structure

```text
imread_benchmark/
  datasets/       package building, FODB selection, GCS materialization
  environments/   descriptor, frozen provisioner, remote tar.zst cache
  plans/          schema-2 plan loading and deterministic expansion
  support/        pre-timing support audits and pinned intersections
  execution/      campaign coordinator, one-run workers, attempts
  artifacts/      atomic bundles and remote commit protocol
  analysis/       canonical loader, statistics, claim gate, publication
  decoders/       entry-point adapters and capability contracts
```

## Contributing

Run the complete local gate before submitting changes:

```bash
uv run pytest -q
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for decoder adapter requirements.

## Citation

```bibtex
@misc{iglovikov2026singlethreadjpegdecoderbenchmarks,
  title={Single-Thread JPEG Decoder Benchmarks Mis-Evaluate ML Data Loaders},
  author={Vladimir Iglovikov},
  year={2026},
  eprint={2605.08731},
  archivePrefix={arXiv},
  primaryClass={cs.PF},
  url={https://arxiv.org/abs/2605.08731}
}
```
