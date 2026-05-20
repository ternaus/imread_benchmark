# Running Benchmarks on Google Cloud

The scripts in `gcp/` are the cloud replication framework for `imread-benchmark`.
They spin up a GCP VM, run the same local `imread-benchmark` CLI against ImageNet
from a GCS bucket, upload JSON results and logs back to GCS, and **delete the VM
when done**. No machine time is wasted. You can start a run, close your laptop,
and fetch results in the morning.

Use this path when you want reproducible CPU comparisons across GCP machine
families, not just a one-off local benchmark.

---

## Prerequisites

### 1. Install Google Cloud SDK

```bash
brew install --cask google-cloud-cli
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

### 2. Upload ImageNet val to GCS (one-time)

Do this once. Every subsequent benchmark run pulls from GCS — no local upload per run.

**Recommended: upload as a single tarball.** Per-file downloads on the VM bottleneck on
GCS small-file API overhead (~50 ms × 50,000 files = 40+ min); a single 5.5 GB tarball
pulls in ~30 sec at same-region GCS bandwidth and is then untarred locally.

```bash
cd ~/data/imagenet            # parent of the val/ directory
tar -cf val.tar val/
gcloud storage cp val.tar gs://my-bucket/imagenet/val.tar
```

`vm_startup.sh` auto-detects `<imagenet-bucket>.tar` and uses the fast path. For
smoke runs (`--smoke`, `--num-images 200`) it untars everything and trims down to N
locally — still faster than per-file fetching even though most of the tarball is
discarded, because hyperdisk-balanced extraction is essentially free.

**Fallback: per-file directory upload** (kept working for existing buckets that
predate the tarball convention):

```bash
./gcp/run.sh \
  --upload-imagenet ~/data/imagenet/val \
  --imagenet-bucket gs://my-bucket/imagenet/val
```

If both `<bucket>/` (per-file dir) and `<bucket>.tar` (single tarball) exist, the
tarball wins. To force the slow path for testing, rename or delete the tarball.

---

## Running Benchmarks

### Start a run and go to bed

```bash
./gcp/run.sh \
  --imagenet-bucket gs://YOUR_BUCKET/imagenet/val \
  --results-bucket  gs://YOUR_BUCKET/imread-results \
  --no-wait
```

This prints a one-liner to fetch results, then exits immediately.
**Close your laptop. The VM runs independently.**

What happens while you sleep:

1. A `c3-standard-8` VM boots on GCP (Intel Sapphire Rapids, 8 vCPU)
2. Downloads ImageNet from GCS to local disk (~30s within GCP, tarball fast path)
3. Runs all supported library benchmarks in memory mode (1-thread + default-thread) — ~15-25 min
4. Runs requested DataLoader benchmarks at 50k images, workers ∈ {0, 2, 4, 8} — ~25-45 min
5. Uploads results to `gs://YOUR_BUCKET/imread-results/<run-id>/output/`
6. **Deletes itself** — no orphaned VM, no ongoing billing

### In the morning

```bash
# Check if done (exit 0 = object exists)
gcloud storage objects describe gs://YOUR_BUCKET/imread-results/<run-id>/DONE

# Fetch results
gcloud storage cp --recursive gs://YOUR_BUCKET/imread-results/<run-id>/output/ ./output/
```

The exact commands are printed by `--no-wait` when you start the run.

### Watch the log mid-run (optional)

```bash
gcloud storage cat gs://YOUR_BUCKET/imread-results/<run-id>/startup.log
```

This streams the full VM stdout, including per-library progress, without SSH.

---

## Stay and wait (blocks until done)

If you want the script to block, download results, and exit all in one command:

```bash
./gcp/run.sh \
  --imagenet-bucket gs://YOUR_BUCKET/imagenet/val \
  --results-bucket  gs://YOUR_BUCKET/imread-results
```

Progress is printed every 30 seconds. Results land in `./output/` when done.
Press Ctrl+C at any point — the VM is deleted immediately via a cleanup trap.

### Run one decoder

Use `--libs` when you only need new data for one decoder. For example, to add
`ajpegli` results without rerunning every library:

```bash
./gcp/run.sh \
  --machine-type c4-standard-16 \
  --smoke \
  --libs ajpegli \
  --imagenet-bucket gs://YOUR_BUCKET/imagenet/val \
  --results-bucket  gs://YOUR_BUCKET/imread-results
```

For the full paper platform matrix:

```bash
./gcp/run-many.sh \
  --machine-types "c4-standard-16 c3d-standard-16 c4d-standard-16 c4a-standard-16 t2a-standard-16" \
  --libs ajpegli \
  --imagenet-bucket gs://YOUR_BUCKET/imagenet/val \
  --results-bucket  gs://YOUR_BUCKET/imread-results
```

---

## Options


| Flag                | Default         | Description                                                          |
| ------------------- | --------------- | -------------------------------------------------------------------- |
| `--imagenet-bucket` | (required)      | GCS path to ImageNet val directory                                   |
| `--results-bucket`  | (required)      | GCS bucket for results and logs                                      |
| `--zone`            | `us-central1-a` | GCP zone                                                             |
| `--machine-type`    | `c3-standard-8` | VM type (Intel Sapphire Rapids, 8 vCPU, 32 GB)                       |
| `--libs`            | `all`           | Comma-separated decoder names, or `all`                              |
| `--num-images`      | `50000`         | Number of images per benchmark run (full ImageNet val)               |
| `--num-runs`        | `5`             | Single-thread timed runs per library (see [Sample size](#sample-size))|
| `--dl-runs`         | `3`             | DataLoader timed runs per worker config                              |
| `--workers`         | `0 2 4 8`       | DataLoader worker counts (drop `1` — torch impl detail vs `0`)       |
| `--no-wait`         | off             | Fire-and-forget; return immediately after VM creation                |
| `--smoke`           | off             | Tiny validation run on a new machine type (~10 min, ~$0.10)          |
| `--no-cache`        | off             | Skip the venv cache entirely (forces a cold install on the VM)       |
| `--force-rebuild`   | off             | Re-resolve PyPI on the VM and reupload the cache (see below)         |
| `--upload-imagenet` | —               | Upload local ImageNet path to `--imagenet-bucket`, then exit         |


All flags can also be set as environment variables (`IMAGENET_BUCKET`, `RESULTS_BUCKET`,
`ZONE`, `MACHINE_TYPE`, `NUM_IMAGES`).

---

## Venv cache

Cold installing decoder libraries on the VM takes ~25 minutes. To avoid paying that on
every run, the first run on a given `(os, arch, deps)` combo tars `venvs/` and uploads it
to `gs://<your-bucket>/imread-cache/`. Subsequent runs pull and extract it in seconds.

Default cache bucket is sibling to your results bucket: if you pass
`--results-bucket gs://my-bucket/imread-results`, the cache lands in
`gs://my-bucket/imread-cache/`. Override with `CACHE_BUCKET=gs://...` or disable per-run
with `--no-cache`.

### Cache key

```
venvs-<os>-<arch>-<sha256(uv.lock)[:12]>.tar.zst
```

Only `uv.lock` is hashed — it already pins every package, version, hash, extra, and index
URL, so no other input adds information. Cosmetic edits to `pyproject.toml`
(formatter, README link, ruff config) do not bust the cache.

### When new library versions ship to PyPI

Cached venvs are **frozen** at whatever `uv.lock` resolved to. New releases on PyPI never
sneak in just because the upstream package shipped — that's the whole point: bit-identical
benchmark inputs across reruns. To pull in newer versions, pick one:

```bash
# Option A: explicit, recorded in git, picked up on the next run automatically.
uv lock --upgrade
git add uv.lock && git commit -m "chore: refresh deps"
./gcp/run.sh ...

# Option B: ad-hoc PyPI re-resolve without editing the lockfile (e.g. monthly
# "is anything faster now" sweep). Bypasses cache lookup, runs `uv pip install`
# fresh on the VM, and OVERWRITES the cached tarball under the same key.
./gcp/run.sh --force-rebuild ...
```

Old cache entries stick around forever in GCS (keyed by old hash), so historical
benchmarks remain reproducible: `git checkout <old-sha> && ./gcp/run.sh ...` rebuilds the
exact env from then.

---

## Cost


| Component                                 | Estimate                       |
| ----------------------------------------- | ------------------------------ |
| VM compute (`c3-standard-8`)              | ~$0.33/hr × 1h ≈ **$0.35**     |
| GCS storage (results, ~5 MB)              | negligible                     |
| GCS egress (ImageNet download within GCP) | free (same region)             |
| **Total per run**                         | **~$0.40** (full benchmark)    |


The VM self-deletes after benchmarks complete, so there is no idle billing.
The 60 GB boot disk is deleted along with the VM.

ImageNet val in GCS costs ~$0.13/month to store. Upload it once, reuse forever.

---

## How It Works

```
LOCAL MACHINE                    GCS BUCKET                    GCP VM
─────────────────────────────────────────────────────────────────────
run.sh packs repo (git archive)
  → uploads repo.tar.gz         [repo.tar.gz]
  → uploads vm_startup.sh       [vm_startup.sh]

gcloud instances create          ←startup-script
(returns immediately)

                                                   VM boots
                                                   vm_startup.sh runs:
                                 [imagenet/val] →  downloads ImageNet
                                                   runs benchmarks
                                 [output/]      ←  uploads results
                                 [DONE]         ←  writes sentinel
                                                   REST API delete call
                                                   VM is gone

run.sh polls [DONE] every 30s
  → finds DONE
  → downloads [output/] to ./output/
  → exits
```

---

## Troubleshooting

**VM vanished without a DONE sentinel**

The VM crashed. Check the log:

```bash
gcloud storage cat gs://my-bucket/imread-results/<run-id>/startup.log
```

**"no active gcloud account" error**

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**A library failed but others succeeded**

`imread-benchmark run` skips failed libraries and continues. Results for successful
libraries are still uploaded. Check the log for `WARN:` lines.

**Want to cancel a running VM**

```bash
gcloud compute instances list   # find the instance name
gcloud compute instances delete imread-benchmark-TIMESTAMP --zone=us-central1-a --quiet
```

**Add Linux-only libraries (`jpeg4py`)**

Already in the decoder registry. `imread-benchmark run` honors the per-decoder
platform-skip metadata, so it automatically participates when you launch on a
matching VM (Linux for jpeg4py).

## Regenerating public and paper assets

After fetching `output/` from GCS, regenerate public README assets with:

```bash
imread-benchmark plot --input output --output docs/assets/benchmarks
imread-benchmark render-readme
```

Publication-style tables and figures use the same JSON outputs but write into the
ignored `_internal/papers/` workspace:

```bash
uv run --extra plot python -m tools.paper_assets --all
```

For the public arXiv preprint workspace specifically:

```bash
uv run --extra plot python -m tools.paper_assets --all --paper-dir _internal/papers/arxiv_preprint
```

Do not commit `_internal/` outputs. They are local manuscript artifacts; the
tracked source of truth is the benchmark JSON plus the generator code.

---

## Sample size

The defaults (`N=50000`, `--num-runs 5`, `--dl-runs 3`, `--workers "0 2 4 8"`)
are picked to give publication-grade precision in the smallest wallclock the
statistics actually justify. Earlier revisions used 50000 × 20 single-thread
runs and 50000 × 5 dataloader runs across `0 1 2 4 8` workers — that was ~4×
slower than necessary for zero gain in any number we cite.

For paper figures and wording, see
[`docs/plotting_and_statistics.md`](plotting_and_statistics.md). In short:
tables report `mean ± sample std`, but comparative claims should use the raw
`raw_throughput_ips` samples plus practical thresholds. Treat DataLoader gaps
below about 5% as top-tier ties unless raw-run uncertainty clearly separates
them.

### Why N = 50000

Full ImageNet val. Two reasons:

1. **No sampling defense in a report.** "We use the standard ImageNet val
   set (N = 50 000)" needs no justification. "We sampled 10 000 images" needs
   a paragraph on selection method, seed, representativeness, and why N is
   below the natural unit.
2. **Doesn't drive wallclock anyway.** Per-run time is `N / throughput`. The
   slow decoders (skimage, imageio at ~200 img/s) take ~4 min/run at N=50k;
   cutting N to 10k saves ~3 min/run, but that's the entire decoder's runtime
   compressed, not the bottleneck. The bottleneck is `N × num_runs`, and
   `num_runs` is the cheaper lever.

### Why num_runs = 5 (down from 20)

With N = 50 000 and per-image σ/μ ≈ 0.4 for JPEG decode time, the standard
error on the *mean throughput* across `N × num_runs` decodes is:

    SE_mean ≈ (σ_per_image / sqrt(N × num_runs)) / μ ≈ 0.4 / sqrt(N × num_runs)

For 50000 × 5 that's ~0.08% relative SE. The interesting effect sizes we
compare are 2× to 100× (decoder throughput differences). Even 1% SE is more
precision than the benchmark decision can use; 0.08% is theatrical.

`num_runs` matters for *per-run percentiles* (p50/p90/p99 across runs), not
for mean precision. With `num_runs = 5`:

- `p50` = median of 5 ≈ stable
- `p90`, `p99` = essentially the max-of-5, no useful resolution

We dropped these columns from publication tables. The headline is `mean ± std img/s`,
and `num_runs = 5` gives a stable std estimate (5 samples is the practical
minimum for a non-degenerate sample variance).

### Why dl-runs = 3

DataLoader runs are dominated by worker spawn variance and the first-batch
warmup (subprocess fork + library re-import). After the first warmup pass
the per-run noise is small; 3 timed reps is enough for `mean ± std`. Going
to 5+ doesn't surface anything that 3 misses.

### Why workers = "0 2 4 8" (dropped `1`)

`num_workers=0` and `num_workers=1` both run the decoder in a single process;
the `0` path skips the worker IPC entirely (in-thread iteration), the `1`
path forks one worker. The wall-clock difference is a torch implementation
detail, not a property of any decoder. Nobody plots it. Keeping `0` (no-fork
baseline), `2` (fork sanity, catches pyvips-on-Arm-style deadlocks), `4`,
and `8` (scaling on 8-vCPU machines).

### If you want the old protocol back

```bash
./gcp/run.sh --num-runs 20 --dl-runs 5 --workers "0 1 2 4 8" ...
```

It's an env-var override, no code change. Useful if you specifically want
percentile data for a separate latency-tail study.

---

## Why no Pillow-SIMD?

Pillow-SIMD was included in the original (2019-2024) revisions of this
benchmark and was dropped 2026-04. Reasons:

1. **Abandoned upstream.** Last release `9.5.0.post2` (May 2023). The repo
   (`uploadcare/pillow-simd`) has had no commits, no issue triage, and no
   wheel uploads since. It is permanently behind mainstream Pillow on every
   security and performance fix.
2. **No Linux wheels.** The project ships only an sdist on PyPI, so every
   install builds from source and pulls in `build-essential`, `zlib1g-dev`,
   `libjpeg-turbo8-dev` at C-compile time. On a cold-cache GCP VM that's
   another ~90 s of apt + ~60 s of compile per run, in exchange for a
   library nobody is shipping.
3. **No industry footprint.** PyPI download stats put `pillow-simd` at
   <0.1% of `pillow`'s volume. None of the modern training stacks
   (PyTorch DataLoader default, torchvision, FFCV, DALI, webdataset) pin
   it; they use either vanilla Pillow or a turbojpeg-backed reader.
4. **Superseded.** The historical SIMD speed-up over vanilla Pillow on
   JPEG decode (~2-3×) is now matched or beaten by `jpeg4py`,
   `simplejpeg`, `PyTurboJPEG`, and `kornia-rs` — all of which are
   actively maintained, ship Linux wheels, and are already in this
   benchmark.
5. **Silent invalidation risk.** Any package that installs vanilla
   `Pillow` into the same venv (e.g. torchvision's transitive pin)
   downgrades pillow-simd in place — the import still says `PIL`, but
   the SIMD intrinsics are gone. The reported number would be a lie
   without obvious warning.

If you specifically need a pillow-simd vs Pillow comparison, the decoder,
dependency group, and CI wiring are preserved in git history — search for
the commit that touches `imread_benchmark/decoders/pillow_simd_decoder.py`
and revert it locally.
