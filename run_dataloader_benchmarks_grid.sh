#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_image_directory> <num_images>"
    echo "Example: $0 ~/data/imagenet/val/ 20000"
    exit 1
fi

DATA_DIR=$1
NUM_IMAGES=$2
NUM_RUNS=5  # Fixed number of runs per configuration

# Arrays of batch sizes and thread counts to test
BATCH_SIZES=(16 32 64 128 256 512 1024 2048 4096)
NUM_THREADS=(5 6 7 12 16)

# Function to format time
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Calculate total number of combinations
TOTAL_COMBINATIONS=$((${#BATCH_SIZES[@]} * ${#NUM_THREADS[@]}))
CURRENT_COMBINATION=0

# Get start time
START_TIME=$(date +%s)

# List of libraries to benchmark
LIBRARIES=("opencv" "pillow" "jpeg4py" "skimage" "imageio" "torchvision" "tensorflow" "kornia")

# Loop through all combinations
for batch_size in "${BATCH_SIZES[@]}"; do
    for num_threads in "${NUM_THREADS[@]}"; do
        CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))

        # Calculate progress and estimated time remaining
        ELAPSED_TIME=$(($(date +%s) - START_TIME))
        if [ $CURRENT_COMBINATION -gt 1 ]; then
            AVG_TIME_PER_COMBINATION=$((ELAPSED_TIME / (CURRENT_COMBINATION - 1)))
            REMAINING_COMBINATIONS=$((TOTAL_COMBINATIONS - CURRENT_COMBINATION + 1))
            ESTIMATED_REMAINING_TIME=$((AVG_TIME_PER_COMBINATION * REMAINING_COMBINATIONS))
            FORMATTED_REMAINING_TIME=$(format_time $ESTIMATED_REMAINING_TIME)
            FORMATTED_ELAPSED_TIME=$(format_time $ELAPSED_TIME)
        else
            FORMATTED_REMAINING_TIME="calculating..."
            FORMATTED_ELAPSED_TIME="00:00:00"
        fi

        echo "============================================================"
        echo "Progress: $CURRENT_COMBINATION/$TOTAL_COMBINATIONS combinations"
        echo "Current configuration: batch_size=$batch_size, num_threads=$num_threads"
        echo "Elapsed time: $FORMATTED_ELAPSED_TIME"
        echo "Estimated time remaining: $FORMATTED_REMAINING_TIME"
        echo "============================================================"

        # Loop through each library
        for lib in "${LIBRARIES[@]}"; do
            echo "Running benchmark for $lib..."
            export BENCHMARK_LIBRARY=$lib
            python imread_benchmark/benchmark_dataloader.py \
                --data-dir "$DATA_DIR" \
                --num-images "$NUM_IMAGES" \
                --num-runs "$NUM_RUNS" \
                --batch-size "$batch_size" \
                --num-workers "$num_threads" \
                --output-dir "output"
        done

        # Optional: add a small delay between runs
        sleep 2
    done
done

# Calculate and display total execution time
TOTAL_TIME=$(($(date +%s) - START_TIME))
FORMATTED_TOTAL_TIME=$(format_time $TOTAL_TIME)
echo "============================================================"
echo "Benchmark grid completed!"
echo "Total execution time: $FORMATTED_TOTAL_TIME"
echo "Results are saved in the output directory organized by batch size and number of workers."
echo "============================================================"
