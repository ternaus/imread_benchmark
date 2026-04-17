#!/usr/bin/env bash
# gcp/run.sh — Launch a GCP VM, run imread benchmarks, fetch results, terminate VM.
#
# Usage:
#   ./gcp/run.sh [options]
#
# Required (or set as env vars):
#   --imagenet-bucket  gs://bucket/path/to/imagenet/val   (or IMAGENET_BUCKET)
#   --results-bucket   gs://bucket/imread-results          (or RESULTS_BUCKET)
#
# Optional:
#   --zone             GCP zone              (default: us-central1-a)
#   --machine-type     GCP machine type      (default: c3-standard-8)
#   --num-images       images to benchmark   (default: 50000)
#   --num-runs         single-thread timed runs per library (default: 20)
#   --dl-runs          DataLoader timed runs per worker config (default: 5)
#   --workers          DataLoader worker counts to test       (default: "0 1 2 4 8")
#   --smoke            short validation run on a new machine type
#                      (forces num-images=2000, num-runs=3, dl-runs=2, workers="0 2 8")
#   --no-wait          fire-and-forget; skip polling + fetch
#   --upload-imagenet  path/to/local/imagenet/val   one-time upload to --imagenet-bucket, then exit
#
# Examples:
#   # One-time: upload ImageNet to GCS
#   ./gcp/run.sh --upload-imagenet ~/imagenet/val --imagenet-bucket gs://my-bucket/imagenet/val
#
#   # Smoke test on a new machine type (~10 min, ~$0.10)
#   ./gcp/run.sh --machine-type c4-standard-16 --smoke \
#     --imagenet-bucket gs://my-bucket/imagenet/val \
#     --results-bucket  gs://my-bucket/imread-results --no-wait
#
#   # Full run (blocks ~4h, fetches results when done)
#   ./gcp/run.sh \
#     --imagenet-bucket gs://my-bucket/imagenet/val \
#     --results-bucket  gs://my-bucket/imread-results
#
#   # Fire and forget
#   ./gcp/run.sh --imagenet-bucket gs://... --results-bucket gs://... --no-wait

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-c3-standard-8}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
NUM_RUNS="${NUM_RUNS:-20}"
DL_RUNS="${DL_RUNS:-5}"
WORKERS="${WORKERS:-0 1 2 4 8}"
IMAGENET_BUCKET="${IMAGENET_BUCKET:-}"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"
NO_WAIT=false
SMOKE=false
NO_CACHE=false
UPLOAD_IMAGENET_LOCAL=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --imagenet-bucket)   IMAGENET_BUCKET="$2";      shift 2 ;;
        --results-bucket)    RESULTS_BUCKET="$2";       shift 2 ;;
        --zone)              ZONE="$2";                 shift 2 ;;
        --machine-type)      MACHINE_TYPE="$2";         shift 2 ;;
        --num-images)        NUM_IMAGES="$2";           shift 2 ;;
        --num-runs)          NUM_RUNS="$2";             shift 2 ;;
        --dl-runs)           DL_RUNS="$2";              shift 2 ;;
        --workers)           WORKERS="$2";              shift 2 ;;
        --no-wait)           NO_WAIT=true;              shift   ;;
        --smoke)             SMOKE=true;                shift   ;;
        --no-cache)          NO_CACHE=true;             shift   ;;
        --upload-imagenet)   UPLOAD_IMAGENET_LOCAL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Smoke mode: smallest run that still validates every library + every
# distinct code path on a new machine type.
#   NUM_IMAGES=200  → ~6 batches at bs=32, exercises the loop without burning time.
#   NUM_RUNS=1      → "did the call succeed?". Stats need a real run.
#   DL_RUNS=1       → same logic for DataLoader rounds.
#   WORKERS="0 2"   → 0 = no-fork path; 2 = fork path (catches pyvips-Arm-style
#                     deadlocks). 4/8 is just scaling, useless for smoke.
if [[ "$SMOKE" == "true" ]]; then
    NUM_IMAGES=200
    NUM_RUNS=1
    DL_RUNS=1
    WORKERS="0 2"
fi

# ── One-time upload subcommand ─────────────────────────────────────────────────
if [[ -n "$UPLOAD_IMAGENET_LOCAL" ]]; then
    if [[ -z "$IMAGENET_BUCKET" ]]; then
        echo "Error: --imagenet-bucket is required for --upload-imagenet"
        exit 1
    fi
    echo "Uploading ImageNet from $UPLOAD_IMAGENET_LOCAL → $IMAGENET_BUCKET"
    echo "This is a one-time operation; subsequent runs reuse the GCS copy."
    # Prefer gcloud storage over gsutil (Google's migration path; avoids macOS multiprocessing noise).
    gcloud storage cp --recursive "$UPLOAD_IMAGENET_LOCAL" "$IMAGENET_BUCKET"
    echo "Upload complete: $IMAGENET_BUCKET"
    exit 0
fi

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$IMAGENET_BUCKET" ]]; then
    echo "Error: --imagenet-bucket is required (or set IMAGENET_BUCKET env var)"
    exit 1
fi
if [[ -z "$RESULTS_BUCKET" ]]; then
    echo "Error: --results-bucket is required (or set RESULTS_BUCKET env var)"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
    echo "Error: no active gcloud account. Run: gcloud auth login"
    exit 1
fi

if ! command -v gcloud &>/dev/null; then
    echo "Error: gcloud not found. Install: brew install --cask gcloud-cli"
    exit 1
fi

# ── Run setup ─────────────────────────────────────────────────────────────────
RUN_NAME="imread-benchmark-$(date +%Y%m%d-%H%M%S)"
RUN_GCS="${RESULTS_BUCKET%/}/${RUN_NAME}"
STARTUP_SCRIPT="$SCRIPT_DIR/vm_startup.sh"

# Sibling cache bucket: same parent as the results bucket, but suffix `-cache`
# so it lives next to per-run artifacts without polluting them. Pre-built
# venvs land here keyed by (os, arch, hash-of-pyproject+uv.lock).
# Override with CACHE_BUCKET env var if you want a separate bucket.
CACHE_BUCKET_DEFAULT="${RESULTS_BUCKET%/*}/imread-cache"
CACHE_BUCKET="${CACHE_BUCKET:-$CACHE_BUCKET_DEFAULT}"
[[ "$NO_CACHE" == "true" ]] && CACHE_BUCKET=""

echo "══════════════════════════════════════════════════════"
echo "  imread benchmark cloud run"
echo "  Run ID       : $RUN_NAME"
echo "  Machine      : $MACHINE_TYPE  ($ZONE)"
echo "  ImageNet     : $IMAGENET_BUCKET"
echo "  Results      : $RUN_GCS"
echo "  Num images   : $NUM_IMAGES"
echo "  Single runs  : $NUM_RUNS"
echo "  DataLoader   : $DL_RUNS runs × workers=[$WORKERS]"
[[ "$SMOKE" == "true" ]]   && echo "  Mode         : SMOKE TEST"
[[ -n "$CACHE_BUCKET" ]]   && echo "  Venv cache   : $CACHE_BUCKET" || echo "  Venv cache   : DISABLED"
echo "  Live log     : gcloud storage cat $RUN_GCS/startup.log"
echo "══════════════════════════════════════════════════════"
echo

# ── Pack and upload repo ──────────────────────────────────────────────────────
echo "[1/4] Packing repo..."
# BSD mktemp (macOS) only substitutes X's that are at the very END of the
# template. Passing `/tmp/imread-repo-XXXXXX.tar.gz` creates the LITERAL file,
# which then collides on the next run. Use a temp dir + fixed filename — the
# dir name is randomised, the contained filename can be anything we want.
REPO_TMPDIR=$(mktemp -d -t imread-repo.XXXXXX)
REPO_TARBALL="$REPO_TMPDIR/repo.tar.gz"
trap 'rm -rf "$REPO_TMPDIR"' EXIT
cd "$REPO_ROOT"
# Capture working tree (committed + uncommitted + untracked-but-not-ignored)
# while excluding artifacts that aren't needed at run time.
# Listing files via git keeps us in sync with .gitignore for untracked files.
# The `[ -e "$f" ]` filter drops files that git still tracks but have been
# deleted-and-not-yet-committed locally — without it, tar bails out with
# "Cannot stat" the moment any tracked file is removed.
{ git ls-files; git ls-files --others --exclude-standard; } \
    | grep -vE '^(output|gcp|docs|_internal|paper.*)/' \
    | while IFS= read -r f; do [ -e "$f" ] && printf '%s\n' "$f"; done \
    | tar -czf "$REPO_TARBALL" -T -

echo "[1/4] Uploading repo + startup script to GCS..."
gcloud --quiet storage cp "$REPO_TARBALL" "$RUN_GCS/repo.tar.gz"
gcloud --quiet storage cp "$STARTUP_SCRIPT" "$RUN_GCS/vm_startup.sh"
# Tarball is in $REPO_TMPDIR which the EXIT trap nukes. No manual rm needed.
echo "[1/4] Done."

# ── Auto-detect arch + disk type from machine type ────────────────────────────
# Arm machines (c4a, t2a) need an arm64 image and hyperdisk-balanced.
# Modern x86 (c4, c4d, n4) need hyperdisk-balanced too.
# Older x86 (c3, c3d, n2, n2d, e2) accept pd-ssd.
case "$MACHINE_TYPE" in
    c4a-*|t2a-*)
        IMAGE_FAMILY=ubuntu-2404-lts-arm64
        BOOT_DISK_TYPE=hyperdisk-balanced
        ;;
    c4-*|c4d-*|n4-*)
        IMAGE_FAMILY=ubuntu-2404-lts-amd64
        BOOT_DISK_TYPE=hyperdisk-balanced
        ;;
    *)
        IMAGE_FAMILY=ubuntu-2404-lts-amd64
        BOOT_DISK_TYPE=pd-ssd
        ;;
esac

echo "  Image family : $IMAGE_FAMILY"
echo "  Boot disk    : $BOOT_DISK_TYPE"
echo

# ── Create VM ─────────────────────────────────────────────────────────────────
echo "[2/4] Creating VM $RUN_NAME..."
gcloud compute instances create "$RUN_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=60GB \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --metadata="results-bucket=$RUN_GCS,imagenet-bucket=$IMAGENET_BUCKET,num-images=$NUM_IMAGES,num-runs=$NUM_RUNS,dl-runs=$DL_RUNS,workers=$WORKERS,repo-tarball=$RUN_GCS/repo.tar.gz,cache-bucket=$CACHE_BUCKET" \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --scopes=storage-rw,compute-rw \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure \
    --quiet
echo "[2/4] VM created. It will self-terminate when benchmarks finish."

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
    echo
    echo "Interrupted — deleting VM $RUN_NAME..."
    gcloud compute instances delete "$RUN_NAME" --zone="$ZONE" --quiet 2>/dev/null || true
    echo "VM deleted. Partial results (if any) may be at $RUN_GCS/output/"
    exit 1
}
trap cleanup INT TERM

# ── Fire-and-forget mode ──────────────────────────────────────────────────────
if [[ "$NO_WAIT" == "true" ]]; then
    echo
    echo "Fire-and-forget mode."
    echo "The VM runs benchmarks, uploads results to GCS, and deletes itself."
    echo "You can close this terminal or shut down your laptop — nothing runs locally."
    echo
    echo "Monitor progress : gcloud storage cat $RUN_GCS/startup.log"
    echo "Check if done    : gcloud storage objects describe $RUN_GCS/DONE"
    echo "Fetch results    : gcloud storage cp --recursive $RUN_GCS/output/ ./output/"
    exit 0
fi

# ── Poll for completion ───────────────────────────────────────────────────────
echo "[3/4] Waiting for benchmarks to finish (polling every 30s)..."
echo "      Live log: gcloud storage cat $RUN_GCS/startup.log"
echo
START_TS=$(date +%s)

while true; do
    sleep 30

    ELAPSED=$(( $(date +%s) - START_TS ))
    ELAPSED_FMT="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"

    if gcloud storage objects describe "$RUN_GCS/DONE" &>/dev/null; then
        echo "[$ELAPSED_FMT] DONE sentinel found — benchmarks complete."
        break
    fi

    # VM self-deletes after writing DONE. If it's gone without DONE, it crashed.
    VM_STATUS=$(gcloud compute instances describe "$RUN_NAME" \
        --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "GONE")

    if [[ "$VM_STATUS" == "GONE" ]]; then
        # Double-check DONE in case sentinel was written just before deletion
        if gcloud storage objects describe "$RUN_GCS/DONE" &>/dev/null; then
            echo "[$ELAPSED_FMT] VM deleted itself after completing. DONE sentinel found."
            break
        fi
        echo "[$ELAPSED_FMT] WARNING: VM is gone but no DONE sentinel. It may have crashed."
        echo "Check logs: gcloud storage cat $RUN_GCS/startup.log"
        exit 1
    fi

    echo "[$ELAPSED_FMT] Status: $VM_STATUS — still running..."
done

# ── Fetch results ─────────────────────────────────────────────────────────────
echo "[4/4] Downloading results..."
mkdir -p "$REPO_ROOT/output"
gcloud storage cp --recursive "$RUN_GCS/output/" "$REPO_ROOT/output/"
echo "[4/4] Results saved to: $REPO_ROOT/output/"

echo
echo "══════════════════════════════════════════════════════"
echo "  Run complete: $RUN_NAME"
echo "  Results    : $REPO_ROOT/output/"
echo "  Full log   : gcloud storage cat $RUN_GCS/startup.log"
echo "  VM self-deleted after benchmarks finished."
echo "══════════════════════════════════════════════════════"
