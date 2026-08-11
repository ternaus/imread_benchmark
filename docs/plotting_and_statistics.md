# Plotting, statistics, and claims

## Canonical inputs

Only validated schema-2 run bundles are evidence. `summary.json` is a derived
cache; the canonical loader preserves every raw timed-pass sample, event,
failure row, runtime probe, and provenance document.

## Statistics

For elapsed seconds, images per second, and microseconds per image, report:

- number of timed-pass samples;
- arithmetic mean;
- median;
- sample standard deviation with `ddof=1`;
- null standard deviation when `n=1`.

The repetition block, not an individual JPEG, is the independent scheduling
unit. When comparing configurations, prefer paired within-block ratios after
matching platform, workload, protocol, output contract, support items, and
repetition.

Do not pool thousands of per-image decodes and present them as thousands of
independent benchmark repetitions. Do not infer a strict winner from small,
noisy differences inside the declared practical margin.

## Claim scope

- `decode-memory` supports statements about resident-byte decoder component
  throughput under the normalized RGB contract.
- `loader-supply` supports statements about supply to a Python consumer under
  the pinned DataLoader configuration.
- neither supports a claim about model training throughput.

The publication claim gate enforces these boundaries.

## Figure selection

Use the smallest figure that answers a declared question:

- paired-ratio or slopegraph for protocol rank change;
- worker curves with raw repetition points for saturation;
- matched workload-transfer plot for native versus mixed FODB;
- exact table for the complete matrix and support coverage.

Every axis includes units. Every caption names platform, workload, protocol,
output contract, support policy, repetitions, and uncertainty statistic.
Avoid truncated axes for bar charts, excessive decimal precision, unlabeled
normalization, and decorative charts without a paper claim.

## Quality and resolution

FODB's natural variants are intentionally heterogeneous. Stratifying them by
megapixels, bits per pixel, quantization-table digest, or quality estimate is a
descriptive workload analysis. It does not identify a causal “quality effect.”

A causal figure needs the controlled resolution × quality package described in
[Experiment design](experiment_design.md), with the encoder, source images,
subsampling, and metadata policy pinned.
