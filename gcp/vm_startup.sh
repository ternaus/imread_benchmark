#!/usr/bin/env bash
# Minimal VM bootstrap. Benchmark semantics live in the schema-2 Python CLI.

set -euo pipefail

METADATA_ROOT="http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADER="Metadata-Flavor: Google"

meta() {
    curl -sf "$METADATA_ROOT/instance/attributes/$1" -H "$METADATA_HEADER"
}

self_delete_vm() {
    local token project instance zone
    token=$(curl -sf "$METADATA_ROOT/instance/service-accounts/default/token" -H "$METADATA_HEADER" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
    project=$(curl -sf "$METADATA_ROOT/project/project-id" -H "$METADATA_HEADER")
    instance=$(curl -sf "$METADATA_ROOT/instance/name" -H "$METADATA_HEADER")
    zone=$(curl -sf "$METADATA_ROOT/instance/zone" -H "$METADATA_HEADER" | sed 's|.*/||')
    curl -sf -X DELETE \
        "https://compute.googleapis.com/compute/v1/projects/$project/zones/$zone/instances/$instance" \
        -H "Authorization: Bearer $token" -o /dev/null
}

RESULTS_STORE=$(meta results-store)
ENVIRONMENT_STORE=$(meta environment-store)
DATASET_STORE=$(meta dataset-store)
DATASET_DESCRIPTOR=$(meta dataset-descriptor)
PLAN_URI=$(meta plan-uri)
REPO_URI=$(meta repo-uri)
REPO_SHA256=$(meta repo-sha256)
RUNNER_REVISION=$(meta runner-revision)
DEPENDENCY_GROUPS=$(meta dependency-groups)
MACHINE_TYPE=$(meta machine-type)
LOCATION=$(meta location)
JOB_ROOT=$(meta job-root)
KEEP_ON_FAILURE=$(meta keep-on-failure)

LOG_FILE=/var/log/imread-benchmark.log
ARTIFACT_ROOT=/opt/imread-job/artifacts
ATTEMPTS_ROOT=/opt/imread-job/attempts
mkdir -p "$ARTIFACT_ROOT" "$ATTEMPTS_ROOT"
exec > >(stdbuf -oL tee -a "$LOG_FILE") 2>&1

sync_diagnostics() {
    gcloud --quiet storage cp "$LOG_FILE" "$JOB_ROOT/startup.log" >/dev/null 2>&1 || true
    if [[ -d "$ATTEMPTS_ROOT" ]]; then
        gcloud --quiet storage rsync --recursive "$ATTEMPTS_ROOT" "$JOB_ROOT/attempts" >/dev/null 2>&1 || true
    fi
}

(
    while true; do
        sleep 15
        sync_diagnostics
    done
) &
SYNC_PID=$!

on_error() {
    local exit_code=$?
    local line=$1
    set +e
    sync_diagnostics
    printf '{"exit_code":%s,"line":%s,"runner_revision":"%s","status":"failed"}\n' \
        "$exit_code" "$line" "$RUNNER_REVISION" \
        | gcloud --quiet storage cp - "$JOB_ROOT/FAILED.json"
    kill "$SYNC_PID" >/dev/null 2>&1 || true
    if [[ "$KEEP_ON_FAILURE" != "1" ]]; then
        self_delete_vm || true
    fi
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR
trap 'kill "$SYNC_PID" >/dev/null 2>&1 || true' EXIT

echo "Starting schema-2 campaign at $(date -u --iso-8601=seconds)"
echo "Runner revision: $RUNNER_REVISION"

export DEBIAN_FRONTEND=noninteractive
cloud-init status --wait || true
apt-get update -q
apt-get install -y -q curl git python3 python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /opt/imread-source
gcloud --quiet storage cp "$REPO_URI" /tmp/repo.tar.gz
echo "$REPO_SHA256  /tmp/repo.tar.gz" | sha256sum --check --strict
tar -xzf /tmp/repo.tar.gz -C /opt/imread-source
cd /opt/imread-source

scripts/install-libjpeg-turbo.sh
LIBJPEG_TURBO_BACKEND=$(/opt/libjpeg-turbo/bin/djpeg -version 2>&1)
export CPATH="/opt/libjpeg-turbo/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="/opt/libjpeg-turbo/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/opt/libjpeg-turbo/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="/opt/libjpeg-turbo/bin:$PATH"
export PKG_CONFIG_PATH="/opt/libjpeg-turbo/lib64/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

UV_PROJECT_ENVIRONMENT=/opt/imread-control uv sync --frozen --no-editable --no-group dev
CONTROL=/opt/imread-control/bin/imread-benchmark

mkdir -p /opt/imread-job /data/datasets /opt/imread-environments
gcloud --quiet storage cp "$PLAN_URI" /opt/imread-job/experiment.yaml
"$CONTROL" dataset materialize "$DATASET_DESCRIPTOR" \
    --store "$DATASET_STORE" \
    --cache-root /data/datasets > /opt/imread-job/dataset.json
PACKAGE_DESCRIPTOR=$(python3 -c 'import json; print(json.load(open("/opt/imread-job/dataset.json"))["descriptor"])')

"$CONTROL" platform capture \
    --output /opt/imread-job/platform.json \
    --cloud-provider gcp \
    --machine-type "$MACHINE_TYPE" \
    --location "$LOCATION" > /opt/imread-job/platform-command.json

IFS=',' read -r -a GROUP_ARRAY <<< "$DEPENDENCY_GROUPS"
for dependency_group in "${GROUP_ARRAY[@]}"; do
    group=$(echo "$dependency_group" | xargs)
    [[ -n "$group" ]]
    "$CONTROL" environment provision \
        --group "$group" \
        --runner-revision "$RUNNER_REVISION" \
        --project-root /opt/imread-source \
        --cache-root /opt/imread-environments \
        --python /usr/bin/python3 \
        --native-backend "libjpeg-turbo=$LIBJPEG_TURBO_BACKEND" \
        --remote-store "$ENVIRONMENT_STORE" > "/opt/imread-job/environment-$group.json"
    ENVIRONMENT_DESCRIPTOR=$(python3 -c \
        "import json; print(json.load(open('/opt/imread-job/environment-$group.json'))['descriptor'])")
    ENVIRONMENT_PYTHON=$(python3 -c \
        "import json; print(json.load(open('/opt/imread-job/environment-$group.json'))['python'])")

    "$ENVIRONMENT_PYTHON" -m imread_benchmark.cli campaign run /opt/imread-job/experiment.yaml \
        --package-descriptor "$PACKAGE_DESCRIPTOR" \
        --environment-descriptor "$ENVIRONMENT_DESCRIPTOR" \
        --platform-descriptor /opt/imread-job/platform.json \
        --artifact-root "$ARTIFACT_ROOT" \
        --attempts-root "$ATTEMPTS_ROOT" \
        --runner-revision "$RUNNER_REVISION" \
        --remote-store "$RESULTS_STORE" \
        --worker-python "$ENVIRONMENT_PYTHON" \
        > "/opt/imread-job/campaign-$group.json"
    sync_diagnostics
done

sync_diagnostics
printf '{"runner_revision":"%s","status":"complete"}\n' "$RUNNER_REVISION" \
    | gcloud --quiet storage cp - "$JOB_ROOT/DONE.json"
kill "$SYNC_PID" >/dev/null 2>&1 || true
self_delete_vm
