# GCP campaigns

The GCP path is a thin VM lifecycle wrapper around the same schema-2 Python CLI
used locally. Shell does not contain decoder matrices, dataset-specific branches,
or result semantics.

## Before launch

1. Build and publish the dataset package.
2. Write a plan that pins its package/workload/manifest IDs.
3. Validate and expand the plan locally.
4. Confirm the target service account can read the private dataset prefix,
   create result/cache objects, and delete its own VM.

## Launch

```bash
./gcp/run.sh \
  --plan experiment.yaml \
  --dataset-store gs://YOUR_BUCKET/imread \
  --dataset-descriptor datasets/<package-id>/package.json \
  --results-store gs://YOUR_BUCKET/imread-results \
  --environment-store gs://YOUR_BUCKET/imread-cache \
  --machine-type c3-standard-16 \
  --zone us-west4-a \
  --groups mainstream \
  --no-wait
```

Use `--groups mainstream,tensorflow` only when the plan contains configurations
for both dependency groups. Each group gets its own frozen environment and
campaign pass; stable run keys prevent overlap.

## VM sequence

1. Install minimal system libraries and `uv`.
2. Download and checksum the exact source snapshot and plan.
3. Install a small non-editable control environment with `uv sync --frozen`.
4. Download the dataset package components, verify every hash and tar member,
   and atomically publish a read-only local cache entry.
5. Capture platform identity and runtime metadata.
6. Restore a content-addressed normalized `tar.zst` frozen worker environment from GCS, or
   build it with `uv sync --frozen --no-editable` and publish the cache.
7. Run pre-timing support audits in fresh processes.
8. Pull already committed run bundles, execute only missing run specs in fresh
   processes, and publish each successful bundle immediately.
9. Upload logs/attempt status, write `DONE.json` last, and delete the VM.

On failure the script uploads `FAILED.json` last and deletes the VM by default.
`--keep-on-failure` retains it for SSH diagnosis and therefore requires manual
cleanup.

## Identity and resume

The launcher computes `runner_revision` from sorted source paths and their file
hashes. It does not use a timestamped tar checksum. Repacking unchanged source
therefore preserves run keys.

A timestamp appears only in the job/attempt name. Canonical bundles live at
stable remote keys:

```text
artifacts/bundles/<bundle-id>/<payload files>
artifacts/runs/<run-key>/COMMITTED.json
```

The commit marker is uploaded with an object-generation create-only precondition.
On restart, only markers with available bundles and valid hashes count as done.
Incomplete prefixes are ignored.

## Multiple machines

```bash
./gcp/run-many.sh \
  --machine-types "c3-standard-16 c3d-standard-16 c4d-standard-16 c4a-standard-16" \
  --plan experiment.yaml \
  --dataset-store gs://YOUR_BUCKET/imread \
  --dataset-descriptor datasets/<package-id>/package.json \
  --results-store gs://YOUR_BUCKET/imread-results \
  --environment-store gs://YOUR_BUCKET/imread-cache
```

The wrapper tries fallback zones only for capacity errors and runs machine types
sequentially. Platform ID is part of every run key, so results cannot collide.

## Fault drill

Before the evidence campaign:

1. launch a smoke plan with at least three run specs;
2. terminate the VM after K committed runs;
3. relaunch the identical source and plan on the same machine type;
4. verify the new campaign reports K skipped and N−K completed runs;
5. validate all local or downloaded bundles with `artifacts validate`.

The test suite exercises the same remote commit, corruption, timeout, process
tree termination, and N−K resume semantics against real subprocesses and a fake
object store. A real x86 and ARM GCP smoke remains a release/evidence gate.
