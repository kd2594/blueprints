#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-nanoGPT.sh

# Use a fixed identifier instead of random UUID to enable reuse
# Change this to a unique name for different projects
RESOURCE_NAME="$(whoami)-nanogpt-shakespeare"

DATASET_NAME="nanoGPT-dataset-${RESOURCE_NAME}"
WORKDIR="dataset-${RESOURCE_NAME}"

mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1

# Download dataset files only if missing
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ]; then
  curl --remote-name-all https://docs.flex.ai/example_data/{train.bin,val.bin}
else
  echo "Dataset files already present locally, skipping download"
fi

# Push dataset only if it doesn't already exist in FlexAI
if ! flexai dataset list | grep -q "${DATASET_NAME}"; then
  flexai dataset push "${DATASET_NAME}" \
    --file train.bin=shakespeare_char/train.bin \
    --file val.bin=shakespeare_char/val.bin
else
  echo "Dataset '${DATASET_NAME}' already exists in FlexAI, skipping push"
fi

# flexai dataset delete "${DATASET_NAME}"

# Start the training run (still use unique UUID for run name to allow multiple runs)
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"
RUN_NAME="quickstart-training-job-${RUN_UUID}"

flexai training run "${RUN_NAME}" \
  --dataset nanoGPT-dataset="${DATASET_NAME}" \
  --repository-url https://github.com/flexaihq/nanogpt \
  --train.py config/train_shakespeare_char.py \
  --dataset_dir="${DATASET_NAME}" \
  --out_dir=/output-checkpoint \
  --max_iters=1500

flexai training inspect "${RUN_NAME}"

#flexai training logs "${RUN_NAME}"

#flexai training checkpoints "${RUN_NAME}"
