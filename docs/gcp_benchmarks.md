# Running Benchmarks on Google Cloud

The scripts in `gcp/` spin up a Linux x86-64 VM on GCP, run all benchmarks against
ImageNet from a GCS bucket, upload results back to GCS, and **delete the VM when done**.
No machine time is wasted. You can start a run, close your laptop, and fetch results
in the morning.

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

```bash
./gcp/run.sh \
  --upload-imagenet ~/imagenet/val \
  --imagenet-bucket gs://my-bucket/imagenet/val
```

Replace `gs://my-bucket/imagenet/val` with your actual bucket path.
The upload is ~6.3 GB and takes a few minutes depending on your connection.

This uses `gcloud storage cp --recursive` (not `gsutil`), which avoids Google’s
deprecation warning and the macOS multiprocessing quirk with `gsutil -m`.

---

## Running Benchmarks

### Start a run and go to bed

```bash
./gcp/run.sh \
  --imagenet-bucket gs://my-bucket/imagenet/val \
  --results-bucket  gs://my-bucket/imread-results \
  --no-wait
```

This prints a one-liner to fetch results, then exits immediately.
**Close your laptop. The VM runs independently.**

What happens while you sleep:

1. A `c3-standard-8` VM boots on GCP (Intel Sapphire Rapids, 8 vCPU)
2. Downloads ImageNet from GCS to local disk (~10s within GCP)
3. Runs all 11 library benchmarks in memory mode (1-thread + default-thread) — ~1h
4. Runs all DataLoader benchmarks at 50k images — ~3h
5. Uploads results to `gs://my-bucket/imread-results/<run-id>/output/`
6. **Deletes itself** — no orphaned VM, no ongoing billing

### In the morning

```bash
# Check if done (exit 0 = object exists)
gcloud storage objects describe gs://my-bucket/imread-results/<run-id>/DONE

# Fetch results
gcloud storage cp --recursive gs://my-bucket/imread-results/<run-id>/output/ ./output/
```

The exact commands are printed by `--no-wait` when you start the run.

### Watch the log mid-run (optional)

```bash
gcloud storage cat gs://my-bucket/imread-results/<run-id>/startup.log
```

This streams the full VM stdout, including per-library progress, without SSH.

---

## Stay and wait (blocks until done)

If you want the script to block, download results, and exit all in one command:

```bash
./gcp/run.sh \
  --imagenet-bucket gs://my-bucket/imagenet/val \
  --results-bucket  gs://my-bucket/imread-results
```

Progress is printed every 30 seconds. Results land in `./output/` when done.
Press Ctrl+C at any point — the VM is deleted immediately via a cleanup trap.

---

## Options


| Flag                | Default         | Description                                                  |
| ------------------- | --------------- | ------------------------------------------------------------ |
| `--imagenet-bucket` | (required)      | GCS path to ImageNet val directory                           |
| `--results-bucket`  | (required)      | GCS bucket for results and logs                              |
| `--zone`            | `us-central1-a` | GCP zone                                                     |
| `--machine-type`    | `c3-standard-8` | VM type (Intel Sapphire Rapids, 8 vCPU, 32 GB)               |
| `--num-images`      | `50000`         | Number of images per benchmark run                           |
| `--no-wait`         | off             | Fire-and-forget; return immediately after VM creation        |
| `--upload-imagenet` | —               | Upload local ImageNet path to `--imagenet-bucket`, then exit |


All flags can also be set as environment variables (`IMAGENET_BUCKET`, `RESULTS_BUCKET`,
`ZONE`, `MACHINE_TYPE`, `NUM_IMAGES`).

---

## Cost


| Component                                 | Estimate                   |
| ----------------------------------------- | -------------------------- |
| VM compute (`c3-standard-8`)              | ~$0.33/hr × 4h ≈ **$1.30** |
| GCS storage (results, ~5 MB)              | negligible                 |
| GCS egress (ImageNet download within GCP) | free (same region)         |
| **Total per run**                         | **~$1.50**                 |


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

`run_benchmarks.sh` skips failed libraries and continues. Results for successful
libraries are still uploaded. Check the log for `WARNING:` lines.

**Want to cancel a running VM**

```bash
gcloud compute instances list   # find the instance name
gcloud compute instances delete imread-benchmark-TIMESTAMP --zone=us-central1-a --quiet
```

**Add Linux-only libraries (`jpeg4py`, `pillow-simd`)**

These libraries are not yet in the decoder registry. Once added, `run_benchmarks.sh`
will automatically run them on Linux. See `imread_benchmark/decoders/` to add them.
