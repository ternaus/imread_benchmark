#!/bin/bash

# Function to show help message
show_help() {
    cat << EOF
Usage: ./run_dataloader_benchmarks.sh <path_to_image_directory> [num_images] [num_runs] [batch_size] [num_workers]

This script runs image reading benchmarks using PyTorch DataLoader for multiple Python libraries.
It creates separate virtual environments for each library and saves results
to output/<operating_system>/<library>_dataloader_results.json

Arguments:
    path_to_image_directory  (Required) Directory containing images to benchmark
    num_images              (Optional) Number of images to process (default: 2000)
    num_runs               (Optional) Number of benchmark runs (default: 20)
    batch_size            (Optional) Batch size for DataLoader (default: 32)
    num_workers          (Optional) Number of worker processes (default: 4)

Example usage:
    # Basic usage with defaults:
    ./run_dataloader_benchmarks.sh ~/dataset/images

    # Custom parameters:
    ./run_dataloader_benchmarks.sh ~/dataset/images 1000 3 64 8

Libraries being benchmarked:
    - opencv (opencv-python-headless)
    - pil (Pillow)
    - skimage (scikit-image)
    - imageio
    - torchvision
    - tensorflow
    - kornia (kornia-rs)

Results will be saved in:
    output/
    ├── linux/          # When run on Linux
    │   ├── opencv_dataloader_results.json
    │   ├── pil_dataloader_results.json
    │   └── ...
    └── darwin/         # When run on macOS
        ├── opencv_dataloader_results.json
        ├── pil_dataloader_results.json
        └── ...
EOF
}

# Show help if -h or --help is passed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Exit on error
set -e

# Base directory for virtual environments
VENV_DIR="venvs"
mkdir -p "$VENV_DIR"

# Create output directory
mkdir -p output

# List of libraries to benchmark
LIBRARIES=("opencv" "pillow" "jpeg4py" "skimage" "imageio" "torchvision" "tensorflow" "kornia" "pillow-simd")

# Function to get libraries based on OS
get_libraries() {
    if [[ "$(uname)" == "Darwin" ]]; then
        # Skip jpeg4py and pillow-simd on macOS
        echo "${LIBRARIES[@]}" | tr ' ' '\n' | grep -v "jpeg4py" | grep -v "pillow-simd" | tr '\n' ' '
    else
        echo "${LIBRARIES[@]}"
    fi
}

# Function to create and activate virtual environment
setup_venv() {
    local lib=$1
    echo "Setting up environment for $lib..."

    # Get the full path to the current Python interpreter
    PYTHON_PATH=$(which python)
    echo "Using Python: $PYTHON_PATH"
    echo "Python version: $($PYTHON_PATH --version)"

    # Create venv with the same Python version
    $PYTHON_PATH -m venv "$VENV_DIR/$lib" --clear

    # Activate virtual environment (works on both Unix and Windows)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        source "$VENV_DIR/$lib/Scripts/activate"
    else
        source "$VENV_DIR/$lib/bin/activate"
    fi

    # Upgrade pip first using the correct Python
    $PYTHON_PATH -m pip install --upgrade pip

    # Install uv using the correct Python
    $PYTHON_PATH -m pip install uv

    # Set UV to use copy mode instead of hardlinks
    export UV_LINK_MODE=copy

    # Install requirements using uv
    uv pip install -U torch albumentations
    uv pip install -r requirements/base.txt
    uv pip install -r "requirements/$lib.txt"
}

# Function to run benchmark for a single library
run_benchmark() {
    local lib=$1
    echo "Running DataLoader benchmark for $lib..."
    export BENCHMARK_LIBRARY=$lib
    python imread_benchmark/benchmark_dataloader.py \
        --data-dir "$DATA_DIR" \
        --num-images "$NUM_IMAGES" \
        --num-runs "$NUM_RUNS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir output
}

# Function to get number of CPU cores
get_cpu_cores() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.ncpu
    else
        nproc
    fi
}


# Check if required arguments are provided
if [ -z "$1" ]; then
    echo "Error: Image directory path is required"
    echo
    show_help
    exit 1
fi

DATA_DIR=$1
NUM_IMAGES=${2:-2000}
NUM_RUNS=${3:-20}
BATCH_SIZE=${4:-32}
NUM_WORKERS=${5:-$(get_cpu_cores)}

echo "Starting DataLoader benchmarks with:"
echo "  Image directory: $DATA_DIR"
echo "  Number of images: $NUM_IMAGES"
echo "  Number of runs: $NUM_RUNS"
echo "  Batch size: $BATCH_SIZE"
echo "  Number of workers: $NUM_WORKERS"
echo

# Run benchmarks for each library
for lib in $(get_libraries); do
    echo "Processing $lib..."
    setup_venv "$lib"
    run_benchmark "$lib"
    deactivate
    echo "Completed $lib"
    echo
done

echo "All DataLoader benchmarks completed!"
echo "Results are saved in the output directory organized by operating system."
echo "Check output/$(uname -s | tr '[:upper:]' '[:lower:]')/ for results."
