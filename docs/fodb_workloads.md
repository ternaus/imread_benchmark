# FODB workload preparation

The FODB builder creates two matched real-world workloads without loading the
full source corpus into RAM:

- `fodb-native`: original camera JPEGs from selected complete scenes;
- `fodb-mixed`: those originals plus Facebook, Instagram, Telegram, Twitter,
  and WhatsApp variants for the same scenes and devices.

FODB is used instead of the 350 GB RAISE RAW collection because this benchmark
measures JPEG decoding. RAW files would first need an arbitrary JPEG encoding
step and would no longer be a real encoded workload.

## Selection

The builder scans ZIP central directories, ignores inspection files, and keeps
only scenes containing the complete declared device × provenance matrix. It
ranks scene IDs by a seed-pinned SHA-256 rule and chooses 12 by default. ZIP
member order therefore cannot change the selection.

For every selected JPEG it verifies the ZIP CRC, computes SHA-256, and records:

- width, height, component count, progressive/baseline status, and subsampling;
- quantization tables and their digest;
- an explicitly labelled IJG-style quality estimate;
- scan count and EXIF/ICC/other metadata sizes;
- compressed bytes and bits per decoded pixel;
- device, scene, and processing provenance.

The quality estimate is not the encoder's original `quality=` setting. Social
services may resize, strip metadata, change subsampling, or use different
quantization logic at once.

## Build

```bash
uv run imread-benchmark dataset fodb-package \
  --archive ~/data/fodb-part01.zip \
  --archive ~/data/fodb-part02.zip \
  --archive ~/data/fodb-part03.zip \
  --output-root ~/data/fodb-benchmark \
  --scene-count 12 \
  --seed 20260729
```

Selected members are extracted once. Native and mixed views use hard links, so
they add directory entries rather than duplicate JPEG payload. The final schema-2
package deduplicates identical content again under `blobs/<sha256>.jpg` and
stores one uncompressed tar plus manifests and a random-access offset index.

## Memory bound

The builder rejects a selected native or mixed workload whose total compressed
bytes exceed `--compressed-byte-limit`. It checks the bound before extracting
JPEGs. A run then reads the complete pinned workload from the package tar into
memory once; logical repeats lengthen timed passes without copying those bytes.
This keeps every timed pass on one identical population instead of combining
separately timed shards.

With `spawn` or `forkserver`, resident JPEG bytes are serialized to each worker,
so the campaign preflight counts `workers + 1` compressed copies. It also counts
prefetched decoded RGB arrays from width/height metadata. The high-resolution
paper plan uses batch size 1 and prefetch factor 1. The broad sweep stops at 8
workers; 12 and 16 are allowed only when the pilot still scales and the complete
conservative estimate fits the plan's RAM fraction.

## Upload

```bash
uv run imread-benchmark dataset publish \
  ~/data/fodb-benchmark/packages/<package-id>/package.json \
  --store gs://YOUR_BUCKET/imread \
  --prefix datasets
```

Keep the GCS prefix private. The command uses create-only object publication;
an existing object with different bytes is a conflict, not an overwrite.
