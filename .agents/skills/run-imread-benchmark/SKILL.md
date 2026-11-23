---
name: run-imread-benchmark
description: Runs, resumes, validates, interprets, or publishes imread_benchmark schema-2 JPEG decoder and DataLoader campaigns locally or on GCP. Use when running, resuming, validating, or publishing imread_benchmark campaigns or interpreting their committed bundles.
---

# Run imread_benchmark

Use this skill for benchmark execution, reruns, resume checks, result
validation, interpretation, or publication. The repository's `README.md`,
`docs/experiment_design.md`, and `docs/reproducing_results.md` are the source of
truth. Do not reconstruct an older command or result format.

## Invariants

- Use only `imread-benchmark list-decoders`, `dataset`, `plan`, `environment`,
  `platform`, `campaign`, `artifacts`, and `publish` commands.
- A timed configuration runs in a fresh subprocess.
- Dataset download, archive verification, support audit, and warmup are outside
  timing.
- Successful bundles are immutable, checksum-validated, and visible only after
  `COMMITTED.json`; timed failures invalidate an attempt.
- Results are comparable only when workload, ordered common-support items,
  protocol, output contract, environment, and platform match.
- Decoder or loader supply is not training speed.

## Local workflow

```bash
uv sync --frozen --group dev --extra mainstream
uv run imread-benchmark list-decoders

uv run imread-benchmark dataset fodb-package \
  --archive ~/data/fodb-part01.zip \
  --archive ~/data/fodb-part02.zip \
  --archive ~/data/fodb-part03.zip \
  --output-root ~/data/fodb-benchmark \
  --scene-count 12 \
  --seed 20260729

uv run imread-benchmark plan instantiate \
  examples/fodb-experiment.template.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --output-dir plans/fodb \
  --workload fodb-native \
  --workload fodb-mixed
uv run imread-benchmark plan validate plans/fodb/fodb-native.yaml \
  --package-descriptor /data/packages/<package-id>/package.json
uv run imread-benchmark plan expand plans/fodb/fodb-native.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --output expanded-plan.json
```

Before launching the full matrix, run the preregistered nine-run
Pillow/OpenCV smoke gate (both protocols, workers 0 and 2, one block) against
`fodb-native`:

```bash
uv run imread-benchmark plan instantiate \
  examples/fodb-smoke.template.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --output-dir plans/smoke \
  --workload fodb-native
```

Run `plans/smoke/fodb-native.yaml` with the `campaign run` command below and
validate its bundles before substituting a full-matrix plan.

Capture platform provenance and provision the frozen worker environment. Commit
every source change first: `REVISION` and the environment cache key cover
`HEAD`, `pyproject.toml`, and `uv.lock`, not uncommitted source edits, so a dirty
worktree can measure stale installed code or reuse results for different code.

```bash
test -z "$(git status --porcelain)" || { echo "commit changes first" >&2; exit 1; }
REVISION=$(git rev-parse HEAD)
uv run imread-benchmark platform capture \
  --output artifacts/platform.json --machine-type local --location local
uv run imread-benchmark environment provision \
  --group mainstream --runner-revision "$REVISION" \
  --project-root . --cache-root .cache/environments
```

Use the emitted descriptor and Python path; pass the smoke plan first, then the
full-matrix plan:

```bash
<environment-python> -m imread_benchmark.cli campaign run plans/smoke/fodb-native.yaml \
  --package-descriptor /data/packages/<package-id>/package.json \
  --environment-descriptor <environment.json> \
  --platform-descriptor artifacts/platform.json \
  --artifact-root artifacts --attempts-root attempts \
  --runner-revision "$REVISION" \
  --worker-python <environment-python>
```

## GCP workflow

Publish the package once, then launch the checked-in ephemeral runner:

```bash
uv run imread-benchmark dataset publish /data/packages/<package-id>/package.json \
  --store gs://BUCKET/imread --prefix datasets

./gcp/run.sh \
  --plan plans/fodb/fodb-native.yaml \
  --dataset-store gs://BUCKET/imread \
  --dataset-descriptor datasets/<package-id>/package.json \
  --results-store gs://BUCKET/imread-results \
  --environment-store gs://BUCKET/imread-cache \
  --machine-type c4-standard-16 --groups mainstream --no-wait
```

The launcher uploads a content-addressed source snapshot and plan. The VM
materializes one sequential dataset payload, restores or builds a frozen
environment, pulls prior committed bundles, runs only missing identities,
checkpoints every committed bundle, writes `DONE` or `FAILED` last, and deletes
itself unless retention was requested.

Launch at most one campaign at a time per results store and plan. Two runners
that start before either commits both treat the same identities as missing; one
create-only commit wins and the other campaign ends `FAILED`. Wait for `DONE` or
`FAILED` before relaunching or resuming the same plan.

## Validate and publish

```bash
uv run imread-benchmark artifacts validate artifacts
uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts --output-dir generated
uv run imread-benchmark publish publication.yaml \
  --artifact-root artifacts --output-dir generated --check
```

Interpret only canonical committed bundles. Preserve raw timed-pass samples,
use paired within-block comparisons when identities permit, report sample
standard deviation with `ddof=1`, and apply the declared practical tie margin.
Never use a manuscript table or historical JSON as the numeric source of truth.

## Paper matrix

Use FODB native and matched mixed workloads for real-world transfer evidence.
Use a separate factorial package for causal resolution or encoder-quality
claims. Build it from pinned lossless PNG sources with `imread-benchmark dataset
controlled-package`; the canonical controls and cell naming are documented in
`docs/controlled_ablation.md`. The broad worker grid is `{0,2,4,8}` with five
randomized repetition blocks. Add `{12,16}` only to cells whose pilot remains
unsaturated or otherwise motivates a targeted second stage.

## Adding a decoder

Use the `add-imread-decoder` skill.
