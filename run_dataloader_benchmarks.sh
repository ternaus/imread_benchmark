#!/bin/bash

show_help() {
    cat << EOF
Usage: ./run_dataloader_benchmarks.sh <path_to_image_directory> [num_images] [num_runs] [workers...]

Measures JPEG decoding throughput inside a PyTorch DataLoader with varying
numbers of worker processes. Images are pre-loaded into memory; only decode
time is measured.

Arguments:
    path_to_image_directory  (Required) Directory containing JPEG images
    num_images               (Optional) Number of images (default: 2000)
    num_runs                 (Optional) Timed rounds per worker count (default: 5)
    workers...               (Optional) num_workers values to test (default: 0 1 2 4 8)

Examples:
    ./run_dataloader_benchmarks.sh ~/data/imagenet/val
    ./run_dataloader_benchmarks.sh ~/data/imagenet/val 2000 5 0 1 2 4 8

System requirements (macOS, install once):
    brew install jpeg-turbo   # simplejpeg, turbojpeg
    brew install vips         # pyvips

Results saved to:
    output/<os>_<cpu>/<library>_dataloader_results.json
EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [ -z "$1" ]; then
    echo "Error: image directory path is required"
    echo
    show_help
    exit 1
fi

set -e

export DYLD_LIBRARY_PATH="/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

DATA_DIR=$1
NUM_IMAGES=${2:-2000}
NUM_RUNS=${3:-5}
# Remaining args are worker counts; default if not provided
if [ $# -ge 4 ]; then
    shift 3
    WORKERS=("$@")
else
    WORKERS=(0 1 2 4 8)
fi

# Share venvs/ with run_benchmarks.sh — torch is now in requirements/base.txt
# so any venv built by the single-thread script can also drive a DataLoader.
VENV_DIR="venvs"
mkdir -p "$VENV_DIR" output

# Excluded from DataLoader benchmarks:
#   tensorflow  — torch + tf in one venv hits numpy/protobuf pin conflicts, and
#                 nobody uses tf.io.decode_jpeg inside a torch DataLoader in
#                 practice (they'd use tf.data).
#   pillow-simd — torchvision's transitive Pillow pin silently downgrades it
#                 back to vanilla Pillow, so the measurement would be a lie.
ALL_LIBRARIES=(
    "opencv"
    "pillow"
    "skimage"
    "imageio"
    "torchvision"
    "kornia"
    "simplejpeg"
    "turbojpeg"
    "imagecodecs"
)

# pyvips on Arm Linux deadlocks PyTorch's default fork start method:
# libvips spawns GLib worker threads at import → fork copies the pthread
# IDs but not the threads → DataLoader workers hang waiting on threads
# that don't exist in the child. The bug is specific to (Linux + aarch64
# + fork). It works on:
#   - x86 Linux (different libvips threadpool init race)
#   - macOS Arm (Python 3.8+ defaults to spawn, not fork)
# Switching torch globally to spawn fixes it but slows every library 3-5×.
# Reported as a finding in the paper instead.
if ! [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
    ALL_LIBRARIES+=("pyvips")
fi

# jpeg4py: Linux-only (no Windows/macOS wheels).
if [[ "$(uname -s)" == "Linux" ]]; then
    ALL_LIBRARIES+=("jpeg4py")
fi

setup_venv() {
    local lib=$1
    # Reuse the venv built by run_benchmarks.sh (single-thread). Only torch needs
    # to be added on top — and only for the libs that go through DataLoader, so
    # tensorflow/pillow-simd/torchvision venvs never see torch and stay clean.
    if [[ -f "$VENV_DIR/$lib/bin/activate" ]]; then
        echo "=== Reusing venv for $lib (adding torch) ==="
        # shellcheck source=/dev/null
        source "$VENV_DIR/$lib/bin/activate"
        export UV_LINK_MODE=copy
        uv pip install -r requirements/dataloader_base.txt
        return 0
    fi
    echo "=== Setting up environment for $lib ==="
    uv venv "$VENV_DIR/$lib" --python python3 --seed
    # shellcheck source=/dev/null
    source "$VENV_DIR/$lib/bin/activate"
    export UV_LINK_MODE=copy
    uv pip install -r requirements/base.txt
    uv pip install -r requirements/dataloader_base.txt
    uv pip install -r "requirements/$lib.txt"
    uv pip install -e . --no-deps
}

run_dataloader_benchmark() {
    local lib=$1
    export BENCHMARK_LIBRARY=$lib
    python imread_benchmark/benchmark_dataloader.py \
        --data-dir "$DATA_DIR" \
        --num-images "$NUM_IMAGES" \
        --num-runs "$NUM_RUNS" \
        --output-dir output \
        --workers "${WORKERS[@]}"
}

echo "Starting DataLoader benchmarks"
echo "  Image directory : $DATA_DIR"
echo "  Number of images: $NUM_IMAGES"
echo "  Number of runs  : $NUM_RUNS"
echo "  Worker counts   : ${WORKERS[*]}"
echo

FAILED=()

for lib in "${ALL_LIBRARIES[@]}"; do
    echo "Processing $lib..."
    if ! setup_venv "$lib"; then
        echo "WARNING: environment setup failed for $lib — skipping"
        FAILED+=("$lib (setup failed)")
        deactivate 2>/dev/null || true
        echo
        continue
    fi

    if ! run_dataloader_benchmark "$lib"; then
        echo "WARNING: benchmark failed for $lib"
        FAILED+=("$lib (benchmark failed)")
    else
        echo "Completed $lib"
    fi

    deactivate 2>/dev/null || true
    echo
done

echo "All DataLoader benchmarks completed!"
echo "Results saved in output/"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "The following libraries were skipped:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
