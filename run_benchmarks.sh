#!/bin/bash

# Function to show help message
show_help() {
    cat << EOF
Usage: ./run_benchmarks.sh <path_to_image_directory> [num_images] [num_runs]

This script runs image reading benchmarks for multiple Python libraries.
It creates separate virtual environments for each library and saves results
to output/<operating_system>/<library>_results.json

Arguments:
    path_to_image_directory  (Required) Directory containing images to benchmark
    num_images              (Optional) Number of images to process (default: 2000)
    num_runs               (Optional) Number of benchmark runs (default: 5)

Example usage:
    # Basic usage with defaults (2000 images, 5 runs):
    ./run_benchmarks.sh ~/dataset/images

    # Custom number of images and runs:
    ./run_benchmarks.sh ~/dataset/images 1000 3

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
    │   ├── opencv_results.json
    │   ├── pil_results.json
    │   └── ...
    └── darwin/         # When run on macOS
        ├── opencv_results.json
        ├── pil_results.json
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

# Function to create and activate virtual environment
setup_venv() {
    local lib=$1
    echo "Setting up environment for $lib..."
    python -m venv "$VENV_DIR/$lib"

    # Activate virtual environment (works on both Unix and Windows)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        source "$VENV_DIR/$lib/Scripts/activate"
    else
        source "$VENV_DIR/$lib/bin/activate"
    fi

    pip install uv

    # Install requirements
    uv pip install -r requirements/base.txt
    uv pip install -r "requirements/$lib.txt"
}

# Function to run benchmark for a single library
run_benchmark() {
    local lib=$1
    echo "Running benchmark for $lib..."
    export BENCHMARK_LIBRARY=$lib
    python imread_benchmark/benchmark_single.py \
        --data-dir "$DATA_DIR" \
        --num-images "$NUM_IMAGES" \
        --num-runs "$NUM_RUNS" \
        --output-dir output
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
NUM_RUNS=${3:-5}

echo "Starting benchmarks with:"
echo "  Image directory: $DATA_DIR"
echo "  Number of images: $NUM_IMAGES"
echo "  Number of runs: $NUM_RUNS"
echo

# Run benchmarks for each library
for lib in "${LIBRARIES[@]}"; do
    echo "Processing $lib..."
    setup_venv "$lib"
    run_benchmark "$lib"
    deactivate
    echo "Completed $lib"
    echo
done

echo "All benchmarks completed!"
echo "Results are saved in the output directory organized by operating system."
echo "Check output/$(uname -s | tr '[:upper:]' '[:lower:]')/ for results."
