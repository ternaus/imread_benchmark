# Experiment design for the strengthened article

## Claims first

The paper should make three bounded claims:

1. Isolated single-thread decode rank is not a reliable proxy for decoder-plus-
   loader supply rank on named CPU platforms and workloads.
2. Decoder recommendations and worker saturation points can change across a
   camera-native high-resolution workload and a realistic social-media-processed
   workload.
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

## Core matrix

Start from [`examples/fodb-experiment.template.yaml`](../examples/fodb-experiment.template.yaml)
and instantiate it once for `fodb-native` and once for `fodb-mixed`, replacing
every package/manifest/count placeholder with the package descriptor values.
The two plans deliberately remain distinct evidence populations.
TensorFlow belongs to its own incompatible dependency group and is not a
PyTorch DataLoader adapter. If its decoder-only result remains useful, run the
separate [`examples/fodb-tensorflow.template.yaml`](../examples/fodb-tensorflow.template.yaml)
as supplementary `decode-memory` evidence; do not insert it into loader curves.

Use the following core GCP platform set at 16 vCPUs:

| Machine type | CPU family | Role in the design |
| --- | --- | --- |
| `c3-standard-16` | Intel Sapphire Rapids | x86 Intel reference |
| `c3d-standard-16` | AMD EPYC Genoa | x86 AMD Zen 4 |
| `c4d-standard-16` | AMD EPYC Turin | matched AMD Zen 5 generation contrast |
| `c4a-standard-16` | Google Axion, Arm Neoverse V2 | current Arm reference |

These assignments follow the current
[Google Cloud machine-family documentation](https://docs.cloud.google.com/compute/docs/general-purpose-machines).
The captured CPU brand, family/model/stepping, architecture, VM type, zone, and
logical CPU count remain the evidence source; the marketing series name is not
used as a substitute for runtime provenance.

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
1 and 6 add cost without answering a primary claim, and 12/16 can make decoded
in-flight memory dominate on large native images.

Use an adaptive second stage only when a pilot shows one of these conditions:

- throughput is still increasing materially at 8 workers;
- the peak is at the tested boundary;
- the curve is non-monotonic enough that its location is unclear;
- runtime worker probes or memory telemetry suggest an oversubscription
  mechanism worth isolating.

Then add `{12,16}` only for the affected decoder × platform × workload cells,
with the same repetition policy. This is a targeted mechanism experiment, not a
post-hoc replacement for the declared core matrix.

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

## Pilot gates before the full campaign

1. Package audit: hashes, manifest counts, resolution/compressed-size strata,
   provenance distribution, and decoded-memory bound.
2. One x86 smoke: Pillow and OpenCV, both protocols, workers 0 and 2.
3. One ARM smoke with the same run specs.
4. Kill after K of N runs; verify a fresh VM launches exactly N−K run workers.
5. Verify every bundle, environment descriptor, platform descriptor, worker
   handshake, and publication sidecar.
6. Freeze the final matrix before launching the full platform sweep.

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
