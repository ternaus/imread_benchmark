#!/usr/bin/env bash
# Run the same schema-2 campaign sequentially across machine types.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_TYPES=""
ZONES="us-west4-a us-central1-a us-central1-c us-east5-b us-east1-b"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine-types) MACHINE_TYPES="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPES="$2"; shift 2 ;;
        --zones) ZONES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --machine-types 'c3-standard-8 c4a-standard-8' [run.sh options]"
            exit 0
            ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "$MACHINE_TYPES" ]]; then
    echo "--machine-types is required" >&2
    exit 2
fi

overall_status=0
for machine_type in $MACHINE_TYPES; do
    launched=false
    for zone in $ZONES; do
        log_file=$(mktemp -t "imread-${machine_type}.XXXXXX")
        set +e
        "$SCRIPT_DIR/run.sh" \
            --machine-type "$machine_type" \
            --zone "$zone" \
            "${PASSTHROUGH[@]}" 2>&1 | tee "$log_file"
        status=${PIPESTATUS[0]}
        set -e
        if [[ $status -eq 0 ]]; then
            launched=true
            rm -f "$log_file"
            break
        fi
        if grep -qE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|does not exist in zone|not available in zone|Invalid value for field.*machineType' "$log_file"; then
            echo "Machine unavailable for $machine_type in $zone; trying the next zone."
            rm -f "$log_file"
            continue
        fi
        echo "Campaign failed for $machine_type in $zone." >&2
        rm -f "$log_file"
        overall_status=1
        break
    done
    if [[ "$launched" != "true" ]]; then
        overall_status=1
    fi
done

exit "$overall_status"
