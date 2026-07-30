# Controlled resolution and JPEG-quality ablation

The article uses two complementary kinds of evidence:

- FODB `fodb-native` and `fodb-mixed` are real workloads with naturally varying
  resolution, quantization tables, metadata, processing service, and compressed
  size. They answer whether recommendations transfer to a realistic mixture.
- The controlled package changes requested long edge and encoder quality while
  holding source pixels, encoder build, chroma subsampling, metadata policy,
  image order, and source membership fixed. It supports bounded mechanism
  claims about those declared factors.

Do not infer the effect of resolution or quality by regressing over FODB's
natural mixture. Those variables are confounded there.

## Source contract

Place the pinned source set in one directory tree as lossless, single-frame PNG
files. The builder rejects JPEG, TIFF, WebP, BMP, animated PNG, transparent PNG,
and any source smaller than the largest requested fixed long edge. This avoids
silently adding an earlier lossy encode, an alpha-compositing policy, upscaling,
or a format-dependent decoder to the experimental design.

The command requires a source dataset name, release, and license, and stores
them in package provenance. It also records every relative path, source byte
hash, normalized RGB pixel hash, dimensions, and source ID. Absolute local
paths do not enter package identity.

## Build all factor cells once

The canonical grid is:

- fixed long edge: 512, 1024, and 2048 pixels;
- a native-size reference level;
- Pillow encoder quality: 50, 75, 90, and 95;
- chroma subsampling: 4:2:0;
- progressive encoding and encoder optimization disabled;
- source metadata stripped; Pillow's fixed container headers retained;
- Pillow LANCZOS resize;
- one seeded source order shared by every cell.

Build the package:

```bash
uv run imread-benchmark dataset controlled-package \
  --source-dir /data/pinned-lossless-png \
  --output-root /data/controlled-jpeg \
  --source-name SOURCE_DATASET_NAME \
  --source-release SOURCE_DATASET_RELEASE \
  --source-license SOURCE_DATASET_LICENSE \
  --source-url SOURCE_DATASET_URL \
  --long-edge 512 \
  --long-edge 1024 \
  --long-edge 2048 \
  --quality 50 \
  --quality 75 \
  --quality 90 \
  --quality 95 \
  --include-native \
  --subsampling 4:2:0 \
  --seed 20260729 \
  --compressed-byte-limit 2147483648
```

The result is one content-addressed package with 16 workloads such as
`controlled-le0512-q050`, `controlled-le2048-q095`, and
`controlled-native-q075`. Each workload contains exactly one generated JPEG per
source ID in the same order. The byte limit applies independently to every
complete resident workload and is checked before packaging.

Pillow and its JPEG backend versions are part of package provenance. The JPEG
bytes and every source are content-addressed, so a different encoder build or
output changes the package ID even if the requested factors are unchanged.

Publish this package to GCS with the same create-only command used for FODB:

```bash
uv run imread-benchmark dataset publish \
  /data/controlled-jpeg/packages/PACKAGE_ID/package.json \
  --store gs://YOUR_BUCKET/imread \
  --prefix datasets
```

## Benchmark and analyze

Generate all 16 workload plans from the package descriptor in one command:

```bash
uv run imread-benchmark plan instantiate \
  examples/controlled-ablation.template.yaml \
  --package-descriptor /data/controlled-jpeg/packages/PACKAGE_ID/package.json \
  --output-dir plans/controlled
```

The command fills each workload's exact manifest ID and item count, then
validates and expands every result before returning it. Keep the platform,
decoder matrix, repetition count, thread profiles, source set, and measurement
settings identical across all 16 plans.

The broad controlled grid measures `decode-memory`; this isolates decoder
capacity without multiplying the already large design by every worker count.
If loader-level follow-up is warranted, predeclare the decoder-selection rule
and run workers `{0, 2, 4, 8}` only for those cells and decoders. Do not describe
that targeted follow-up as a new exhaustive sweep.

Use complete fresh-process repetition blocks as the independent samples. Pair
cells by platform, repetition, decoder/thread profile, and the recorded source
set. Report requested quality as an encoder setting, not as a universal JPEG
quality scale. The fixed 512/1024/2048 levels support a resolution contrast;
`native` is a matched real-size reference whose pixel dimensions still vary by
source.

Acceptable claim:

> Under the pinned Pillow encoder build, fixed 4:2:0 subsampling, and matched
> source images, changing the requested long edge from 512 to 2048 changed the
> measured decode-throughput ratio between decoders A and B on platform P.

Unacceptable claim:

> JPEG quality generally causes decoder A to beat decoder B.

The latter exceeds one encoder, one subsampling policy, the tested factor
levels, and the named platforms.
