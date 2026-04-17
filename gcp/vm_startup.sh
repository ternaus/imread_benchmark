#!/usr/bin/env bash
# gcp/vm_startup.sh — Runs automatically on the GCP VM at boot via instance metadata.
# DO NOT run this locally. It is uploaded to GCS and referenced as the startup-script.
#
# Flow:
#   1. Read config from GCP instance metadata
#   2. Install system deps + uv
#   3. Download repo tarball from GCS
#   4. Download ImageNet val from GCS to local disk
#   5. Run single-thread + default-thread benchmarks
#   6. Run DataLoader benchmarks
#   7. Upload results to GCS
#   8. Write DONE sentinel
#   9. Self-delete the VM instance (requires compute-rw scope)

set -euo pipefail

# Show the failing line if anything dies — vital for debugging via serial console.
trap 'echo "ERROR: vm_startup.sh failed at line $LINENO (exit $?)" >&2' ERR

# google-metadata-script-runner runs the script as root with a minimal env.
# HOME is not set, which breaks `set -u` when uv / other tools reference it.
export HOME="${HOME:-/root}"

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
META_HEADER="Metadata-Flavor: Google"

meta() {
    curl -sf "$METADATA_URL/$1" -H "$META_HEADER"
}

# ── Read metadata ──────────────────────────────────────────────────────────────
RESULTS_BUCKET=$(meta results-bucket)
IMAGENET_BUCKET=$(meta imagenet-bucket)
NUM_IMAGES=$(meta num-images)
REPO_TARBALL=$(meta repo-tarball)
# Optional metadata with sensible defaults so older callers still work.
NUM_RUNS=$(meta num-runs 2>/dev/null || echo 20)
DL_RUNS=$(meta dl-runs 2>/dev/null || echo 5)
WORKERS=$(meta workers 2>/dev/null || echo "0 1 2 4 8")
CACHE_BUCKET=$(meta cache-bucket 2>/dev/null || echo "")

LOG_FILE="/var/log/imread_benchmark.log"
GCS_LOG="$RESULTS_BUCKET/startup.log"

# Redirect all output to a local log file (line-buffered).
# The lazy "one gsutil cp at EOF" approach doesn't give a live log;
# instead, spawn a background loop below that uploads the log every 15s.
exec > >(stdbuf -oL tee -a "$LOG_FILE") 2>&1

# Background log-sync: uploads the log to GCS every 15 seconds so the user
# can `gcloud storage cat ...` at any time and see recent progress.
# PATH may not yet include gcloud/gsutil during early boot — fall back gracefully.
(
    while true; do
        sleep 15
        if command -v gcloud >/dev/null 2>&1; then
            gcloud --quiet storage cp "$LOG_FILE" "$GCS_LOG" >/dev/null 2>&1 || true
        elif command -v gsutil >/dev/null 2>&1; then
            gsutil -q cp "$LOG_FILE" "$GCS_LOG" 2>/dev/null || true
        fi
    done
) &
LOG_SYNC_PID=$!

# Background output-sync: pushes /opt/imread_benchmark/output/ to GCS every
# 30s. Without this, a hang/crash mid-run (e.g. pyvips fork-deadlock on Arm)
# loses every JSON we already produced and forces an SSH rescue. With it,
# the latest finished library is always recoverable. Uses `gcloud storage rsync`
# so we only ship newly-written files, not the whole tree every cycle.
OUTPUT_DIR="/opt/imread_benchmark/output"
GCS_OUTPUT="$RESULTS_BUCKET/output"
(
    while true; do
        sleep 30
        if [[ -d "$OUTPUT_DIR" ]] && command -v gcloud >/dev/null 2>&1; then
            gcloud --quiet storage rsync --recursive \
                "$OUTPUT_DIR" "$GCS_OUTPUT" >/dev/null 2>&1 || true
        fi
    done
) &
OUTPUT_SYNC_PID=$!

trap 'kill $LOG_SYNC_PID $OUTPUT_SYNC_PID 2>/dev/null || true' EXIT

echo "═══════════════════════════════════════════════════"
echo "  imread benchmark — VM startup"
echo "  $(date -u)"
echo "  Results bucket : $RESULTS_BUCKET"
echo "  ImageNet       : $IMAGENET_BUCKET"
echo "  Num images     : $NUM_IMAGES"
echo "  Repo tarball   : $REPO_TARBALL"
echo "═══════════════════════════════════════════════════"

# ── System deps ────────────────────────────────────────────────────────────────
echo
echo "[step 1] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Wait for cloud-init / unattended-upgrades to release the apt lock.
# Without this, `apt-get update` races with boot-time updates and exits non-zero.
echo "  Waiting for cloud-init to finish (releases apt lock)..."
cloud-init status --wait || true

# Wait until the dpkg lock is gone (belt-and-suspenders).
for i in $(seq 1 60); do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
       && ! fuser /var/lib/apt/lists/lock >/dev/null 2>&1; then
        break
    fi
    echo "  apt locked (attempt $i/60), sleeping 5s..."
    sleep 5
done

# Run apt without piping through tail — pipefail would mask the real error.
# Retry update once in case of transient mirror flake.
apt-get update -qq || apt-get update -qq
apt-get install -y -qq \
    libjpeg-turbo8-dev \
    libturbojpeg0-dev \
    libvips-dev \
    curl \
    git \
    python3 \
    python3-pip \
    zstd
echo "[step 1] Done."

# ── uv ─────────────────────────────────────────────────────────────────────────
echo
echo "[step 2] Installing uv..."
curl -Ls https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
echo "[step 2] Done."

# ── Repo ───────────────────────────────────────────────────────────────────────
echo
echo "[step 3] Downloading + extracting repo..."
gcloud --quiet storage cp "$REPO_TARBALL" /tmp/repo.tar.gz
mkdir -p /opt/imread_benchmark
tar -xzf /tmp/repo.tar.gz -C /opt/imread_benchmark
cd /opt/imread_benchmark
echo "[step 3] Done. Contents:"
ls -1

# ── ImageNet ───────────────────────────────────────────────────────────────────
# Only fetch the N images we actually need. Saves ~5-10 min and several GB
# on smoke runs (e.g. 2k images instead of 50k). For full 50k runs this is
# the same volume as a recursive copy but parallelized via `gcloud storage cp -I`.
echo
echo "[step 4] Downloading $NUM_IMAGES images from $IMAGENET_BUCKET ..."
IMAGENET_DIR=/data/imagenet/val
mkdir -p "$IMAGENET_DIR"

# Strip any trailing slash from the bucket path so `**` glob behaves consistently.
BUCKET_PREFIX="${IMAGENET_BUCKET%/}"

# Materialize the full sorted list to disk first, THEN take the first N.
# Doing `... | sort | head -n N` would SIGPIPE upstream stages once head
# closes stdin; with `set -o pipefail` that kills the whole script (exit 141).
gcloud --quiet storage ls "$BUCKET_PREFIX/**" \
    | grep -Ei '\.(jpe?g)$' \
    | sort > /tmp/imagenet_all.txt
head -n "$NUM_IMAGES" /tmp/imagenet_all.txt > /tmp/imagenet_files.txt

WANTED=$(wc -l < /tmp/imagenet_files.txt)
echo "  Selected $WANTED files; downloading in parallel..."
# `-I` reads source URIs from stdin; gcloud parallelizes downloads internally.
time gcloud --quiet storage cp -I "$IMAGENET_DIR/" < /tmp/imagenet_files.txt

IMAGE_COUNT=$(find "$IMAGENET_DIR" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
echo "[step 4] Done. $IMAGE_COUNT JPEGs in $IMAGENET_DIR"

# ── Venv cache pull ────────────────────────────────────────────────────────────
# Skip the ~25-min `uv pip install` × 11 libs by pulling pre-built venvs from
# GCS, keyed by (os, arch, hash-of-pyproject+uv.lock). Cache is populated at the
# end of a successful run, so the first run on a new (machine_arch, deps) combo
# is full-cost; every subsequent run reuses it in seconds.
echo
echo "[cache] Computing venv cache key..."
CACHE_HIT=false

if [[ -n "$CACHE_BUCKET" ]]; then
    # Hash anything that affects the resolved venv contents.
    # uv.lock captures the full resolved set; pyproject.toml captures extras
    # + index config. Together they uniquely identify what venvs/ should look like.
    REQ_HASH=$(cat pyproject.toml uv.lock 2>/dev/null \
               | sha256sum | cut -c1-12)
    CACHE_KEY="venvs-$(uname -s)-$(uname -m)-${REQ_HASH}.tar.zst"
    CACHE_PATH="${CACHE_BUCKET%/}/${CACHE_KEY}"
    echo "[cache] Key  : $CACHE_KEY"
    echo "[cache] Path : $CACHE_PATH"

    if gcloud storage objects describe "$CACHE_PATH" >/dev/null 2>&1; then
        echo "[cache] HIT — downloading + extracting venvs..."
        time gcloud --quiet storage cp "$CACHE_PATH" /tmp/venvs.tar.zst
        time tar --use-compress-program='zstd -d' -xf /tmp/venvs.tar.zst
        rm -f /tmp/venvs.tar.zst
        CACHE_HIT=true
        echo "[cache] Restored $(find venvs -mindepth 1 -maxdepth 1 -type d | wc -l) venvs."
    else
        echo "[cache] MISS — will build venvs and populate cache after the run."
    fi
else
    echo "[cache] DISABLED (no cache-bucket metadata)"
fi

# ── Control-plane venv (just to get the `imread-benchmark` CLI) ────────────────
# A small venv (numpy + pandas + typer) that drives the per-group worker venvs.
# Worker venvs (mainstream/tensorflow/pillow-simd) are created lazily by the CLI.
# Idempotent: cached `venvs/control/` from a previous run is reused as-is, since
# `uv pip install -e .` is fast on a populated venv and picks up any code changes.
echo
echo "[step 5] Setting up control-plane venv..."
if [[ ! -x venvs/control/bin/python ]]; then
    uv venv venvs/control --python python3 --seed
fi
# shellcheck source=/dev/null
source venvs/control/bin/activate
UV_LINK_MODE=copy uv pip install -e .
echo "[step 5] Done."

# ── Run benchmarks via the unified CLI ─────────────────────────────────────────
# `imread-benchmark run --mode both` orchestrates:
#   - venv setup per group (mainstream / tensorflow / pillow-simd)
#   - single + default-thread benchmark for each lib
#   - DataLoader benchmark for each lib eligible on this platform
# Platform skips (jpeg4py off macOS, pyvips off Arm-Linux DataLoader, etc.) are
# encoded on the decoder classes themselves, no shell `case` games.
echo
echo "[step 6] Running benchmarks..."
echo "         NUM_IMAGES=$NUM_IMAGES  NUM_RUNS=$NUM_RUNS  DL_RUNS=$DL_RUNS"
echo "         WORKERS=$WORKERS"
WORKERS_CSV=$(echo "$WORKERS" | tr ' ' ',')
imread-benchmark run \
    --data-dir "$IMAGENET_DIR" \
    --output-dir output \
    --libs all \
    --mode both \
    --num-images "$NUM_IMAGES" \
    --num-runs "$NUM_RUNS" \
    --dataloader-runs "$DL_RUNS" \
    --workers "$WORKERS_CSV"
echo "[step 6] Done."

# ── Upload results ─────────────────────────────────────────────────────────────
echo
echo "[step 7] Final flush of results to $RESULTS_BUCKET/output/ ..."
# The background rsync above ships results every 30s, so this is mostly
# a no-op. Keeping it as a last-mile guarantee in case a JSON was written
# in the final 30s window before completion.
gcloud --quiet storage rsync --recursive output "$RESULTS_BUCKET/output"
echo "[step 7] Done."

# ── Venv cache push (only on cold-cache run) ───────────────────────────────────
# Tar venvs/ and upload so the next run on this (os, arch, deps) combo
# can skip ~25 min of `uv pip install`. Skip if we hit the cache (no point
# re-uploading what we already pulled).
if [[ -n "$CACHE_BUCKET" && "$CACHE_HIT" != "true" ]]; then
    echo
    echo "[cache] Populating venv cache at $CACHE_PATH ..."
    # zstd -3 = good speed/ratio tradeoff. -T0 = use all cores.
    time tar --use-compress-program='zstd -3 -T0' -cf /tmp/venvs.tar.zst venvs/
    echo "[cache] Tarball size: $(du -h /tmp/venvs.tar.zst | cut -f1)"
    time gcloud --quiet storage cp /tmp/venvs.tar.zst "$CACHE_PATH"
    rm -f /tmp/venvs.tar.zst
    echo "[cache] Done."
fi

# ── DONE sentinel ──────────────────────────────────────────────────────────────
echo
echo "[step 8] Writing DONE sentinel..."
date -u | gcloud --quiet storage cp - "$RESULTS_BUCKET/DONE"
echo "[step 8] Done."

echo
echo "═══════════════════════════════════════════════════"
echo "  All benchmarks complete. $(date -u)"
echo "  Deleting this VM instance."
echo "═══════════════════════════════════════════════════"

# Flush log to GCS one final time before self-deletion
sleep 2
gcloud --quiet storage cp "$LOG_FILE" "$GCS_LOG" >/dev/null 2>&1 || true

# Self-delete via GCP Compute REST API.
# Requires --scopes=compute-rw on instance creation (set in gcp/run.sh).
_METADATA="http://metadata.google.internal/computeMetadata/v1"
_TOKEN=$(curl -sf "$_METADATA/instance/service-accounts/default/token" \
    -H "Metadata-Flavor: Google" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
_PROJECT=$(curl -sf "$_METADATA/project/project-id"    -H "Metadata-Flavor: Google")
_INSTANCE=$(curl -sf "$_METADATA/instance/name"        -H "Metadata-Flavor: Google")
_ZONE=$(curl -sf     "$_METADATA/instance/zone"        -H "Metadata-Flavor: Google" | sed 's|.*/||')

curl -sf -X DELETE \
    "https://compute.googleapis.com/compute/v1/projects/$_PROJECT/zones/$_ZONE/instances/$_INSTANCE" \
    -H "Authorization: Bearer $_TOKEN" \
    -o /dev/null
