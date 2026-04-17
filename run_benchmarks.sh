#!/bin/bash

show_help() {
    cat << EOF
Usage: ./run_benchmarks.sh <path_to_image_directory> [num_images] [num_runs] [mode]

Runs JPEG decoding benchmarks for multiple Python libraries on ImageNet validation images.
Each library gets an isolated virtual environment to avoid dependency conflicts.

Arguments:
    path_to_image_directory  (Required) Directory containing JPEG images
    num_images               (Optional) Number of images to use (default: 2000)
    num_runs                 (Optional) Number of timed benchmark runs (default: 20)
    mode                     (Optional) 'memory' or 'disk' (default: memory)

Examples:
    ./run_benchmarks.sh ~/imagenet/val
    ./run_benchmarks.sh ~/imagenet/val 2000 20 memory

System requirements (macOS, install once):
    brew install jpeg-turbo   # required by simplejpeg, turbojpeg
    brew install vips         # required by pyvips

Libraries benchmarked:
    opencv, pillow, skimage, imageio, torchvision, tensorflow,
    kornia, simplejpeg, turbojpeg, imagecodecs, pyvips

Results are saved to:
    output/<os>_<cpu>/
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

# Ensure Homebrew-installed native libraries (libvips, libjpeg-turbo, …) are
# visible to venvs created from non-Homebrew Python interpreters (e.g. miniconda).
export DYLD_LIBRARY_PATH="/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

DATA_DIR=$1
NUM_IMAGES=${2:-2000}
NUM_RUNS=${3:-20}
MODE=${4:-memory}
VENV_DIR="venvs"

mkdir -p "$VENV_DIR" output

# Cross-platform base set.
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

# Platform-specific additions.
# jpeg4py: ctypes bindings to libjpeg-turbo, works on any Linux (x86 + Arm).
# pillow-simd: x86-only fork of Pillow with SSE/AVX intrinsics — won't build on Arm.
case "$(uname -s)" in
    Linux)
        ALL_LIBRARIES+=("jpeg4py")
        if [[ "$(uname -m)" == "x86_64" ]]; then
            ALL_LIBRARIES+=("pillow-simd")
        fi
        ;;
esac

# Pre-flight: warn about missing brew deps that pip cannot provide.
check_brew_deps() {
    local missing=()
    if ! brew list --formula jpeg-turbo &>/dev/null 2>&1; then
        missing+=("jpeg-turbo  # needed by simplejpeg, turbojpeg")
    fi
    if ! brew list --formula vips &>/dev/null 2>&1; then
        missing+=("vips        # needed by pyvips")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo "WARNING: the following brew packages are not installed."
        echo "The libraries that depend on them will be skipped."
        echo ""
        for dep in "${missing[@]}"; do
            echo "  brew install $dep"
        done
        echo ""
    fi
}

setup_venv() {
    local lib=$1
    # Reuse a pre-built venv if present (e.g. restored from GCS cache by the
    # caller). Skips ~2 min of `uv pip install` per library.
    if [[ -f "$VENV_DIR/$lib/bin/activate" ]]; then
        echo "=== Reusing existing venv for $lib ==="
        # shellcheck source=/dev/null
        source "$VENV_DIR/$lib/bin/activate"
        return 0
    fi
    echo "=== Setting up environment for $lib ==="
    uv venv "$VENV_DIR/$lib" --python python3 --seed
    # shellcheck source=/dev/null
    source "$VENV_DIR/$lib/bin/activate"
    export UV_LINK_MODE=copy
    uv pip install -r requirements/base.txt
    uv pip install -r "requirements/$lib.txt"
    uv pip install -e . --no-deps
}

run_benchmark() {
    local lib=$1
    local num_threads=$2  # 0 = library default, 1 = single-threaded
    export BENCHMARK_LIBRARY=$lib
    python imread_benchmark/benchmark_single.py \
        --data-dir "$DATA_DIR" \
        --num-images "$NUM_IMAGES" \
        --num-runs "$NUM_RUNS" \
        --output-dir output \
        --mode "$MODE" \
        --num-threads "$num_threads"
}

echo "Starting benchmarks"
echo "  Image directory : $DATA_DIR"
echo "  Number of images: $NUM_IMAGES"
echo "  Number of runs  : $NUM_RUNS"
echo "  Mode            : $MODE"
echo

check_brew_deps

FAILED=()

get_default_threads() {
    local lib=$1
    BENCHMARK_LIBRARY=$lib python - <<'EOF'
import os
from imread_benchmark.decoders import REGISTRY
lib = os.environ["BENCHMARK_LIBRARY"]
decoder = REGISTRY[lib]()
print(decoder.get_num_threads())
EOF
}

for lib in "${ALL_LIBRARIES[@]}"; do
    echo "Processing $lib..."
    if ! setup_venv "$lib"; then
        echo "WARNING: environment setup failed for $lib — skipping"
        FAILED+=("$lib (setup failed)")
        deactivate 2>/dev/null || true
        echo
        continue
    fi

    # Single-threaded run (always)
    echo "  [1 thread]"
    if ! run_benchmark "$lib" 1; then
        echo "WARNING: single-thread benchmark failed for $lib"
        FAILED+=("$lib (1-thread run failed)")
        deactivate 2>/dev/null || true
        echo
        continue
    fi
    echo "  Completed $lib (1 thread)"

    # Default-threads run — skip if library is inherently single-threaded
    default_threads=$(get_default_threads "$lib")
    if [ "$default_threads" -le 1 ]; then
        echo "  [default threads = 1, copying 1t result — no second run needed]"
        system_id=$(BENCHMARK_LIBRARY="$lib" python -c "from imread_benchmark.utils import get_system_identifier; print(get_system_identifier())")
        src="output/${system_id}/${lib}_1t_results.json"
        dst="output/${system_id}/${lib}_${default_threads}t_results.json"
        # src and dst are the same filename when default_threads==1, nothing to do
        [ "$src" != "$dst" ] && cp "$src" "$dst"
    else
        echo "  [library default: $default_threads threads]"
        if ! run_benchmark "$lib" 0; then
            echo "WARNING: default-thread benchmark failed for $lib"
            FAILED+=("$lib (default-thread run failed)")
        else
            echo "  Completed $lib ($default_threads threads)"
        fi
    fi

    deactivate 2>/dev/null || true
    echo
done

echo "All benchmarks completed!"
echo "Results saved in output/"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "The following libraries were skipped:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
