#!/usr/bin/env bash
# run-many.sh — drive ./gcp/run.sh sequentially across a list of machine types
# with stockout retry and network-tolerant polling.
#
# Why this exists, given run.sh already has a polling loop:
#   1. run.sh's polling treats a failed `gcloud compute instances describe`
#      as "VM disappeared" → exits 1. A 30s home-WiFi blip kills it. The
#      VM keeps running in the cloud, finishes, self-deletes — but the
#      local script is dead and the next iteration of a wrapper for-loop
#      would launch a new VM possibly while the old one's quota is still
#      held. With a 32 vCPU project quota that's a real wedge.
#   2. GCP zonal stockouts (ZONE_RESOURCE_POOL_EXHAUSTED) happen randomly
#      for c4-* / c4d-* / c4a-* — the modern families on hyperdisk-balanced
#      live in a smaller capacity pool than c3 / n2d. We retry across zones
#      automatically instead of you babysitting at 2am.
#   3. If run.sh fails to even create the VM (stockout, quota, image bug),
#      there's nothing to poll — but a naive wrapper that extracts RUN_GCS
#      from the banner before checking exit code will spin forever waiting
#      for a DONE/FAILED that no VM will ever write.
#
# Usage:
#   ./gcp/run-many.sh \
#     --machine-types "c4-standard-16 c3d-standard-16 c4a-standard-16" \
#     --imagenet-bucket gs://my-bucket/imagenet/val \
#     --results-bucket  gs://my-bucket/imread-results \
#     [--zones "us-central1-a us-central1-c us-east5-b"] \
#     [--smoke]
#
# Anything not listed above is forwarded verbatim to ./gcp/run.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/run.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
MACHINE_TYPES=""
ZONES_FALLBACK="us-central1-a us-central1-c us-east5-b us-east1-b us-west1-a"
POLL_SECS=60
PASSTHROUGH=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine-types) MACHINE_TYPES="$2"; shift 2 ;;
        --zones)         ZONES_FALLBACK="$2"; shift 2 ;;
        --poll-secs)     POLL_SECS="$2"; shift 2 ;;
        # `--zone X` from the user becomes the FIRST fallback zone — same
        # mental model as run.sh ("try this zone first") without the
        # downstream confusion of having two `--zone` flags.
        --zone)          ZONES_FALLBACK="$2 $ZONES_FALLBACK"; shift 2 ;;
        # `--machine-type X` is the single-machine shortcut.
        --machine-type)  MACHINE_TYPES="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "$MACHINE_TYPES" ]]; then
    echo "ERROR: --machine-types is required (space-separated list)." >&2
    exit 2
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# extract_run_gcs <run-log-file>
# Pull the RUN_GCS bucket path from a captured run.sh banner. Returns empty
# string if not found (which means VM creation failed before the banner
# reached the "Results:" line — handled by the caller).
extract_run_gcs() {
    grep -oE 'gs://[^ ]+/imread-benchmark-[0-9-]+' "$1" 2>/dev/null \
        | head -n 1
}

# is_stockout <gcloud-error-output>
# True when the failure is "this zone is temporarily out of capacity for
# this machine type" — those are worth retrying in another zone. Quota
# errors / IAM errors / image errors are NOT — those need a human.
is_stockout() {
    grep -qE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources' "$1"
}

# poll_until_terminal <run-gcs>
# Network-tolerant polling. All gcloud failures are silently swallowed —
# transient blips just become "another POLL_SECS of waiting" rather than a
# fatal local exit. Returns 0 on DONE, 1 on FAILED, 2 on timeout (24h).
poll_until_terminal() {
    local run_gcs="$1"
    local start_ts
    start_ts=$(date +%s)
    local timeout_secs=$((24 * 3600))

    while true; do
        sleep "$POLL_SECS"
        local elapsed=$(( $(date +%s) - start_ts ))
        if (( elapsed > timeout_secs )); then
            echo "  [poll] timed out after 24h — bailing." >&2
            return 2
        fi

        if gcloud --quiet storage objects describe "$run_gcs/DONE" \
            >/dev/null 2>&1; then
            echo "  [poll] DONE after $((elapsed / 60))m"
            return 0
        fi
        if gcloud --quiet storage objects describe "$run_gcs/FAILED" \
            >/dev/null 2>&1; then
            echo "  [poll] FAILED after $((elapsed / 60))m"
            echo "         $(gcloud --quiet storage cat "$run_gcs/FAILED" 2>/dev/null | head -1)"
            return 1
        fi

        printf '  [poll] %dm elapsed, no terminal sentinel yet...\n' \
            "$((elapsed / 60))"
    done
}

# ── Main loop ─────────────────────────────────────────────────────────────────
overall_rc=0

for M in $MACHINE_TYPES; do
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  Machine type: $M"
    echo "════════════════════════════════════════════════════════════════"

    run_log="$(mktemp -t "imread-runlog-${M}.XXXXXX")"
    trap 'rm -f "$run_log"' EXIT

    # ── Try each fallback zone until one boots a VM ───────────────────────
    booted_zone=""
    for Z in $ZONES_FALLBACK; do
        echo "  → trying zone $Z..."
        # Use --no-wait so creation is decoupled from polling. We do our
        # OWN polling below with the network-tolerant loop.
        if "$RUN_SH" \
            --machine-type "$M" \
            --zone "$Z" \
            --no-wait \
            "${PASSTHROUGH[@]}" 2>&1 | tee "$run_log"; then
            booted_zone="$Z"
            break
        fi

        # Distinguish "no capacity here, try next zone" from "real config bug"
        if is_stockout "$run_log"; then
            echo "  ⚠ stockout in $Z, trying next zone..."
            continue
        fi
        echo "  ✗ run.sh failed in $Z for non-stockout reason — skipping $M." >&2
        cat "$run_log" >&2
        overall_rc=1
        break
    done

    if [[ -z "$booted_zone" ]]; then
        echo "  ✗ exhausted all zones for $M — skipping." >&2
        overall_rc=1
        continue
    fi

    # ── Find the run's GCS path and poll ──────────────────────────────────
    run_gcs="$(extract_run_gcs "$run_log")"
    if [[ -z "$run_gcs" ]]; then
        echo "  ✗ could not parse RUN_GCS from run.sh output for $M — skipping poll." >&2
        overall_rc=1
        continue
    fi
    echo "  RUN_GCS: $run_gcs"
    echo "  Polling for DONE/FAILED (every ${POLL_SECS}s, network-tolerant)..."

    # Don't let `set -e` kill the wrapper if poll returns FAILED or timeout —
    # we want to fetch whatever partial results landed and move to the next
    # machine type.
    set +e
    poll_until_terminal "$run_gcs"
    poll_rc=$?
    set -e

    # ── Fetch results regardless of DONE vs FAILED ────────────────────────
    # Even FAILED runs may have uploaded partial output/ from the ERR trap
    # in vm_startup.sh (which flushes before writing the FAILED sentinel).
    # rsync to ./output/ — corrects the nesting bug present in run.sh's
    # original `cp --recursive`.
    echo "  Fetching results..."
    if ! gcloud --quiet storage rsync --recursive \
        "$run_gcs/output/" "./output/"; then
        echo "  ⚠ rsync failed (network?). Retry later with:" >&2
        echo "     gcloud storage rsync -r $run_gcs/output/ ./output/" >&2
        overall_rc=1
    fi

    if (( poll_rc != 0 )); then
        overall_rc=1
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
if (( overall_rc == 0 )); then
    echo "  All machines completed successfully."
else
    echo "  Completed with errors — see warnings above."
fi
echo "════════════════════════════════════════════════════════════════"
exit "$overall_rc"
