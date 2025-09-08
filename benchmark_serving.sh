#!/bin/bash

BACKEND="${1:-lmdeploy}"
DATASET_PATH="/nvme1/shared/ShareGPT_V3_unfiltered_cleaned_split.json"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# install dependencies for the benchmark_serving.py script
python3 -m pip install aiohttp numpy transformers

echo "Using backend: $BACKEND"

CONFIGS=(
    "--dataset-name sharegpt --num-prompts 10000"
    "--dataset-name random --num-prompts 2000 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0"
    "--dataset-name random --num-prompts 500 --random-input-len 2000 --random-output-len 6000 --random-range-ratio 1.0"
    "--dataset-name random --num-prompts 500 --random-input-len 6000 --random-output-len 2000 --random-range-ratio 1.0"
)

for i in "${!CONFIGS[@]}"; do
    echo "Running benchmark ${CONFIGS[$i]} with backend: $BACKEND"

    python ${SCRIPT_DIR}/benchmark_serving.py \
        --backend "$BACKEND" \
        --dataset-path "$DATASET_PATH" \
        ${CONFIGS[$i]}
    echo "----------------------------------------"
done

echo "All benchmarks completed for backend: $BACKEND"
