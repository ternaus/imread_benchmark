#!/usr/bin/env bash
# Launch one ephemeral GCP worker for a schema-2 benchmark campaign.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ZONE="${ZONE:-us-west4-a}"
MACHINE_TYPE="${MACHINE_TYPE:-c3-standard-16}"
BOOT_DISK_GB="${BOOT_DISK_GB:-150}"
DEPENDENCY_GROUPS="${IMREAD_DEPENDENCY_GROUPS:-mainstream}"
PLAN_PATH=""
DATASET_STORE=""
DATASET_DESCRIPTOR=""
RESULTS_STORE=""
ENVIRONMENT_STORE=""
DOWNLOAD_DIR=""
NO_WAIT=false
KEEP_ON_FAILURE=false

usage() {
    sed -n '2,34p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) PLAN_PATH="$2"; shift 2 ;;
        --dataset-store) DATASET_STORE="$2"; shift 2 ;;
        --dataset-descriptor) DATASET_DESCRIPTOR="$2"; shift 2 ;;
        --results-store) RESULTS_STORE="$2"; shift 2 ;;
        --environment-store) ENVIRONMENT_STORE="$2"; shift 2 ;;
        --groups) DEPENDENCY_GROUPS="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --boot-disk-gb) BOOT_DISK_GB="$2"; shift 2 ;;
        --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
        --no-wait) NO_WAIT=true; shift ;;
        --keep-on-failure) KEEP_ON_FAILURE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

for required in PLAN_PATH DATASET_STORE DATASET_DESCRIPTOR RESULTS_STORE; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required option for $required" >&2
        exit 2
    fi
done
if [[ ! -f "$PLAN_PATH" ]]; then
    echo "Plan does not exist: $PLAN_PATH" >&2
    exit 2
fi
if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required" >&2
    exit 127
fi
ENVIRONMENT_STORE="${ENVIRONMENT_STORE:-$RESULTS_STORE}"

case "$MACHINE_TYPE" in
    c4a-*) IMAGE_FAMILY=ubuntu-2404-lts-arm64; BOOT_DISK_TYPE=hyperdisk-balanced ;;
    t2a-*) IMAGE_FAMILY=ubuntu-2404-lts-arm64; BOOT_DISK_TYPE=pd-balanced ;;
    c4-*|c4d-*|n4-*) IMAGE_FAMILY=ubuntu-2404-lts-amd64; BOOT_DISK_TYPE=hyperdisk-balanced ;;
    *) IMAGE_FAMILY=ubuntu-2404-lts-amd64; BOOT_DISK_TYPE=pd-ssd ;;
esac

TEMP_ROOT=$(mktemp -d -t imread-gcp.XXXXXX)
cleanup_temp() {
    rm -rf "$TEMP_ROOT"
}
trap cleanup_temp EXIT

cd "$REPO_ROOT"
{
    git ls-files
    git ls-files --others --exclude-standard
} | sort -u | grep -E '^(imread_benchmark/|pyproject\.toml$|uv\.lock$|README\.md$|LICENSE)' > "$TEMP_ROOT/source-files.txt"

RUNNER_REVISION=$(python3 - "$TEMP_ROOT/source-files.txt" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    path = pathlib.Path(line)
    if not path.is_file():
        continue
    digest.update(line.encode())
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)

REPO_ARCHIVE="$TEMP_ROOT/repo.tar.gz"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$REPO_ARCHIVE" -T "$TEMP_ROOT/source-files.txt"
REPO_ARCHIVE_SHA256=$(shasum -a 256 "$REPO_ARCHIVE" | awk '{print $1}')
PLAN_SHA256=$(shasum -a 256 "$PLAN_PATH" | awk '{print $1}')
CONTROL_ROOT="${RESULTS_STORE%/}/control"
REPO_URI="$CONTROL_ROOT/sources/$RUNNER_REVISION/$REPO_ARCHIVE_SHA256.tar.gz"
PLAN_URI="$CONTROL_ROOT/plans/$PLAN_SHA256.yaml"

upload_create_only() {
    local source=$1
    local destination=$2
    if gcloud storage objects describe "$destination" >/dev/null 2>&1; then
        return
    fi
    gcloud --quiet storage cp "$source" "$destination" --if-generation-match=0
}

upload_create_only "$REPO_ARCHIVE" "$REPO_URI"
upload_create_only "$PLAN_PATH" "$PLAN_URI"

RUN_NAME="imread-$(date +%Y%m%d-%H%M%S)-${RUNNER_REVISION:0:8}"
JOB_ROOT="${RESULTS_STORE%/}/jobs/$RUN_NAME"
KEEP_VALUE=0
[[ "$KEEP_ON_FAILURE" == "true" ]] && KEEP_VALUE=1
METADATA="results-store=$RESULTS_STORE,environment-store=$ENVIRONMENT_STORE,dataset-store=$DATASET_STORE,dataset-descriptor=$DATASET_DESCRIPTOR,plan-uri=$PLAN_URI,repo-uri=$REPO_URI,repo-sha256=$REPO_ARCHIVE_SHA256,runner-revision=$RUNNER_REVISION,dependency-groups=$DEPENDENCY_GROUPS,machine-type=$MACHINE_TYPE,location=$ZONE,job-root=$JOB_ROOT,keep-on-failure=$KEEP_VALUE"

echo "Run              : $RUN_NAME"
echo "Runner revision  : $RUNNER_REVISION"
echo "Machine          : $MACHINE_TYPE ($ZONE)"
echo "Dataset          : $DATASET_STORE/$DATASET_DESCRIPTOR"
echo "Results          : $RESULTS_STORE"
echo "Environment cache: $ENVIRONMENT_STORE"
echo "Groups           : $DEPENDENCY_GROUPS"
echo "Live log         : gcloud storage cat $JOB_ROOT/startup.log"

gcloud compute instances create "$RUN_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --metadata="$METADATA" \
    --metadata-from-file=startup-script="$SCRIPT_DIR/vm_startup.sh" \
    --scopes=storage-rw,compute-rw \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure \
    --quiet

delete_on_interrupt() {
    gcloud compute instances delete "$RUN_NAME" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
    exit 130
}
trap delete_on_interrupt INT TERM

if [[ "$NO_WAIT" == "true" ]]; then
    echo "VM is running asynchronously and will delete itself."
    exit 0
fi

while true; do
    sleep 20
    if gcloud storage objects describe "$JOB_ROOT/DONE.json" >/dev/null 2>&1; then
        echo "Campaign worker completed."
        break
    fi
    if gcloud storage objects describe "$JOB_ROOT/FAILED.json" >/dev/null 2>&1; then
        gcloud storage cat "$JOB_ROOT/FAILED.json" || true
        echo "Campaign worker failed; see $JOB_ROOT/startup.log" >&2
        exit 1
    fi
    if ! gcloud compute instances describe "$RUN_NAME" --zone="$ZONE" >/dev/null 2>&1; then
        echo "VM disappeared without a terminal marker; see $JOB_ROOT/startup.log" >&2
        exit 1
    fi
done

if [[ -n "$DOWNLOAD_DIR" ]]; then
    mkdir -p "$DOWNLOAD_DIR"
    gcloud storage rsync --recursive "$JOB_ROOT" "$DOWNLOAD_DIR/$RUN_NAME"
fi
