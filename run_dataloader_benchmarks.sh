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

VENV_DIR="venvs_dataloader"
mkdir -p "$VENV_DIR" output

ALL_LIBRARIES=(
    "opencv"
    "pillow"
    "skimage"
    "imageio"
    "torchvision"
    "tensorflow"
    "kornia"
    "simplejpeg"
    "turbojpeg"
    "imagecodecs"
    "pyvips"
)

setup_venv() {
    local lib=$1
    echo "=== Setting up DataLoader environment for $lib ==="
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
