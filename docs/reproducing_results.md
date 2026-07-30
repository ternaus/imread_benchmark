# Reproducing publication artifacts

The repository intentionally contains no historical result matrix. A publication
artifact must start from a frozen set of schema-2 bundles produced by the named
plan, dataset package, environments, source revision, and platforms.

Reproduction uses the committed `uv.lock`; do not upgrade dependencies while
reproducing a campaign. The one-time latest-version freeze happens before the
first pilot and is documented in [Experiment design](experiment_design.md).
Each environment descriptor records the lock hash, exact installed
distributions, and declared native backend versions.

## Validate evidence

```bash
uv sync --frozen --group dev
uv run imread-benchmark artifacts validate artifacts
```

Validation checks the exact file set, every payload hash and size, bundle and run
identity, package/support/config IDs, ordered items, sample indices and counts,
runtime thread/worker probes, required phase events, environment/platform
descriptors, and the complete summary recomputed from raw samples.

Before a campaign starts, plan validation pins its timeout and maximum RAM
fraction. Campaign preflight rejects configurations whose conservative resident
compressed-data plus decoded-prefetch estimate exceeds that fraction of the
captured platform memory.

## Publication specification

Example:

```yaml
schema_version: "2.0"
claim_scope: loader-supply
filters:
  config.protocol_id: loader-supply
  dataset.workload_id: fodb-mixed
statistic: images_per_second
practical_margin_percent: 5
generator_revision: <40-or-64-character-source-revision>
```

Generate:

```bash
uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts \
  --output-dir generated
```

Verify byte-for-byte freshness in CI:

```bash
uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts \
  --output-dir generated \
  --check
```

`results.json` contains exact selected bundle rows and raw timed-pass values,
plus configuration groups aggregated across complete repetition blocks.
Group-level `n` is the number of independent run blocks; `raw_run_means` keeps
the corresponding per-run means and sample standard deviation uses `ddof=1`.
`provenance.json` records every bundle ID/run key, filters, claim scope,
generator revision, practical margin, result hash, and publication ID.

Do not copy summary values manually into another plotting script. Figures and
tables should consume this checked publication output or the canonical bundle
loader with an equivalent provenance sidecar.
