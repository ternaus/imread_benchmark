# Experiment design for the strengthened article

## Claims first

The paper should make three bounded claims:

1. Isolated single-thread decode rank is not a reliable proxy for decoder-plus-
   loader supply rank on named CPU platforms and workloads.
2. Decoder ranks and worker saturation points can change across a camera-native
   high-resolution workload and a realistic social-media-processed workload.
3. Observed loader supply is only a component measurement and does not imply a
   training-speed difference without measured consumer demand or an end-to-end
   training protocol.

The framework rejects a training claim when only `decode-memory` or
`loader-supply` bundles are selected.

## Dataset regimes

Use matched FODB scenes and devices so content changes less than it would across
unrelated corpora.

| Regime | Purpose | Interpretation |
| --- | --- | --- |
| `fodb-native` | Large original camera JPEGs | High-resolution real-workload evidence |
| `fodb-mixed` | Original plus five platform-processed variants | Realistic mixture of sizes, quantization tables, compression ratios, and metadata |
| Controlled resolution × quality package | Mechanism ablation on pinned source images | Causal effect of the chosen re-encoding factors within this encoder |

The first two are real workload comparisons. Resolution, processing service,
encoder implementation, quantization, metadata, and compression ratio change
together. Report JPEG `quality_estimate` only as an estimate derived from
quantization tables, never as the source encoder's true quality parameter.

For a causal sentence such as “larger resolution changes decoder X relative to
decoder Y,” build the separate package from pinned lossless source images with
the canonical `dataset controlled-package` command. It declares and records:

- long edge: 512, 1024, 2048, and native;
- encoder quality: 50, 75, 90, and 95;
- fixed chroma subsampling and metadata policy;
- identical source image IDs in every cell.

Do not merge those rows with FODB and call the union a single resolution effect.
The full build and interpretation contract is in
[Controlled resolution and JPEG-quality ablation](controlled_ablation.md).

## Dependency freeze

Use the latest stable dependency set available at the preregistered freeze,
then keep it unchanged across every pilot and evidence run. The 2026-07-30
freeze resolves the direct mainstream decoder dependencies as follows on the
Python 3.12 benchmark image:

| Distribution | Frozen version |
| --- | ---: |
| ajpegli | 1.0.0 |
| imagecodecs | 2026.6.26 |
| imageio | 2.37.4 |
| jpeg4py (Linux only) | 0.1.4 |
| kornia-rs | 0.1.14 |
| opencv-python-headless | 5.0.0.93 |
| Pillow | 12.3.0 |
| PyTurboJPEG | 2.5.0 |
| pyvips | 3.1.1 |
| pyvips-binary | 8.18.4 |
| scikit-image | 0.26.0 |
| simplejpeg | 1.9.0 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |

PyTurboJPEG uses the verified upstream libjpeg-turbo 3.2.0 build on both
x86-64 and Arm. Transitive dependencies and Python-version-specific branches
are authoritative in `uv.lock`, rather than duplicated in this document.

Create a new freeze before the first smoke, not between machines:

```bash
uv lock --upgrade
uv sync --frozen --group dev --extra mainstream
uv run pytest -q
uv run pre-commit run --all-files
```

Commit `pyproject.toml`, `uv.lock`, adapter changes, and the native installer
together. The environment identity includes the lock hash, exact installed
distribution versions, and declared native backend versions. If a dependency
release is accepted after a smoke has produced evidence, discard that evidence
and restart every platform with the new environment identity. Otherwise defer
the release until the next benchmark revision.

## Core matrix

Start from [`examples/fodb-experiment.template.yaml`](../examples/fodb-experiment.template.yaml)
and generate the two pinned plans directly from the package descriptor:

```bash
uv run imread-benchmark plan instantiate \
  examples/fodb-experiment.template.yaml \
  --package-descriptor /data/packages/PACKAGE_ID/package.json \
  --output-dir plans/fodb \
  --workload fodb-native \
  --workload fodb-mixed
```

The command fills the package, workload, manifest, and item-count identities,
then validates and expands each result before returning it. The two plans
deliberately remain distinct evidence populations and are reused unchanged
across CPU platforms.
TensorFlow belongs to its own incompatible dependency group and is not a
PyTorch DataLoader adapter. If its decoder-only result remains useful, run the
separate [`examples/fodb-tensorflow.template.yaml`](../examples/fodb-tensorflow.template.yaml)
as supplementary `decode-memory` evidence; do not insert it into loader curves.

Use the following core GCP platform set at 16 vCPUs:

| Machine type | CPU family | Role in the design |
| --- | --- | --- |
| `c4-standard-16` | Intel Xeon (runtime CPU captured) | x86 Intel reference |
| `c3d-standard-16` | AMD EPYC Genoa | x86 AMD Zen 4 |
| `c4d-standard-16` | AMD EPYC Turin | matched AMD Zen 5 generation contrast |
| `c4a-standard-16` | Google Axion, Arm Neoverse V2 | current Arm reference |

These assignments follow the current
[Google Cloud machine-family documentation](https://docs.cloud.google.com/compute/docs/general-purpose-machines).
The captured CPU brand, family/model/stepping, architecture, VM type, zone, and
logical CPU count remain the evidence source; the marketing series name is not
used as a substitute for runtime provenance. Zone is execution provenance, not
a platform-identity factor: capacity may move a run between zones. A platform
group requires matching captured machine type, CPU identity, architecture, and
logical CPU count. Publication output keeps every raw platform ID and zone, so
the grouping is inspectable and a zone-specific anomaly remains visible.

For each named CPU platform and each real workload:

- all supported mainstream decoders under `normalized-rgb`;
- `decode-memory` with decoder default threads and, only for adapters with
  verified thread control, an explicitly requested setting;
- `loader-supply` with workers `{0, 2, 4, 8}`;
- batch size 1 and prefetch factor 1 for high-resolution FODB, persistent
  workers, and an explicit start method;
- five randomized repetition blocks for the broad matrix;
- one complete timed traversal per fresh-process block; the five blocks, rather
  than repeated passes inside one process, are the independent repetitions;
- every timed pass long enough to exceed the preregistered minimum duration,
  using logical repeats rather than copying JPEG bytes.

Thread profiles belong to each decoder in the plan, not to the protocol as a
whole. Plan expansion rejects a requested thread count when the adapter cannot
set and report it, before any cloud VM is launched.

The dense sweep `{0,1,2,4,6,8,12,16}` is not part of the broad matrix. Workers
1 and 6 add cost without answering a primary claim, and high worker counts can
make decoded in-flight memory dominate on large native images.

Use an adaptive second stage only when a pilot shows one of these conditions:

- throughput is still increasing materially at 8 workers;
- the peak is at the tested boundary;
- the curve is non-monotonic enough that its location is unclear;
- runtime worker probes or memory telemetry suggest an oversubscription
  mechanism worth isolating.

Extend one boundary at a time. First add `12` only for the affected decoder ×
platform × workload cells, with the same repetition policy. This is a targeted
mechanism experiment, not a post-hoc replacement for the declared core matrix.

Freeze each follow-up stage before observing its new worker count. A cell enters
the `workers=12` stage when its controlled-thread loader configuration has its
highest observed mean at `workers=8` over the completed `{0,2,4,8}` matrix. The
stage adds five fresh-process repetitions at `workers=12` and leaves every
other cell unchanged. Before launching a possible `workers=16` stage, freeze a
material and repeat-consistent stopping rule: the mean at 12 must exceed the
mean at 8 by at least 5%, and all five paired 12/8 block ratios must be at
least one. This avoids spending a second extension on a numerically maximal but
practically flat boundary. It is an operational follow-up rule, not a
statistical-significance claim.

The completed FODB matrix selects 87 of 96 controlled decoder × platform ×
workload cells for the first extension, for 435 additional bundles. The
checked-in follow-up templates encode the frozen `workers=12` selection:

| Workload and platform | Eligible decoders | Template |
| --- | ---: | --- |
| FODB-mixed, all four platforms | 12 | `fodb-worker-followup-all.template.yaml` |
| FODB-native, AMD Zen 5 and Axion | 12 | `fodb-worker-followup-all.template.yaml` |
| FODB-native, Intel 8581C | 8 | `fodb-worker-followup-native-intel.template.yaml` |
| FODB-native, AMD Zen 4 | 7 | `fodb-worker-followup-native-zen4.template.yaml` |

The completed 12-worker stage selected 58 configurations for 16 workers, or
290 fresh-process bundles: all 48 FODB-mixed cells, eight Intel-native cells,
and the two Axion-native cells for `ajpegli` and Pillow. Native Zen 4 and Zen 5
cells did not meet the rule. The corresponding checked-in templates are
`fodb-worker-followup-w16-mixed.template.yaml`,
`fodb-worker-followup-w16-native-intel.template.yaml`, and
`fodb-worker-followup-w16-native-axion.template.yaml`.

Neither stage turns the naturally heterogeneous FODB workloads into a causal
test of resolution or JPEG quality.

Every plan declares `execution.maximum_memory_fraction`. Campaign preflight
conservatively counts resident compressed-byte replicas (including `spawn` or
`forkserver` copies), prefetched decoded RGB batches, the consumer batch, and
one decoder temporary per worker. A configuration that exceeds the declared
fraction of measured platform RAM is rejected before support audit or timing.
The package's compressed-byte cap bounds each complete resident workload; it
does not split one logical run into separately timed shards.

## Support and failures

Run support audits before timing. Preserve two views:

- operational support: all items accepted by one decoder/context;
- common support: the ordered intersection for a declared comparison group.

Headline decoder comparisons use common support. Report operational coverage
and every excluded item separately. A timed failure invalidates the run; it is
not converted into a faster partial traversal.

## Repetitions and uncertainty

The unit of repetition is a complete randomized run block, not an individual
image. Keep every timed-pass sample. Report the mean, median, sample standard
deviation (`ddof=1`), and `n`; for `n=1`, sample standard deviation is null.

Use paired within-block ratios when comparing configurations that share
platform, workload, support items, protocol, output contract, and repetition.
Treat differences inside the declared practical margin as practically tied.
Do not turn overlapping noisy means into a strict ranking sentence.

When a capacity retry moves a run to another zone, retain the zone and complete
platform descriptor with that run. The run remains in the same platform group
only when the captured non-zone identity matches. Do not make a zone-level
performance claim from that grouping.

## Pilot gates before the full campaign

1. Package audit: hashes, manifest counts, resolution/compressed-size strata,
   provenance distribution, and decoded-memory bound.
2. One x86 smoke from
   [`examples/fodb-smoke.template.yaml`](../examples/fodb-smoke.template.yaml):
   Pillow and OpenCV, both protocols, workers 0 and 2, exactly nine run specs.
3. One ARM smoke with the same run specs.
4. Kill after K of N runs; verify a fresh VM launches exactly N−K run workers.
5. Verify every bundle, environment descriptor, platform descriptor, worker
   handshake, and publication sidecar.
6. Freeze the final matrix before launching the full platform sweep.

Instantiate the smoke plan against `fodb-native` before launching either
machine:

```bash
uv run imread-benchmark plan instantiate \
  examples/fodb-smoke.template.yaml \
  --package-descriptor /data/packages/PACKAGE_ID/package.json \
  --output-dir plans/smoke \
  --workload fodb-native
```

## Figures and tables

Build figures only after selecting the claim they support:

- protocol rank-change slopegraph or paired-ratio plot;
- worker curves for representative inversion/saturation cases, with raw
  repetition points and uncertainty;
- workload transfer plot comparing native and mixed common-support ratios;
- exact appendix matrices for every platform/workload/configuration;
- coverage table for operational/common support and failures.

Every caption names protocol, workload, platform, output contract, support
policy, repetition count, and uncertainty statistic. Raw samples and bundle IDs
remain machine-readable in publication output.
