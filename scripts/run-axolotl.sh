#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-ultralytics-yolo.sh

HF_TOKEN="paste-ur-token-here"

# Check if HF_TOKEN is set and not empty
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please export your Hugging Face token as HF_TOKEN. https://huggingface.co/docs/hub/en/security-tokens"
  exit 1
fi


# Use a fixed identifier instead of random UUID to enable reuse
# Change this to a unique name for different projects
RESOURCE_NAME="$(whoami)-qwen2-axolotl-french"

# Check if secret already exists, create only if not
if ! flexai secret list | grep -q "hf-token-${RESOURCE_NAME}"; then
  echo "Creating HF token secret..."
  echo "$HF_TOKEN" | flexai secret create "hf-token-${RESOURCE_NAME}"
else
  echo "HF token secret already exists, skipping creation"
fi

# Check if storage connection already exists, create only if not
if ! flexai storage list | grep -q "hf-conn-${RESOURCE_NAME}"; then
  echo "Creating HF storage connection..."
  flexai storage create "hf-conn-${RESOURCE_NAME}" \
    --provider huggingface \
    --hf-token-name "hf-token-${RESOURCE_NAME}"
else
  echo "HF storage connection already exists, skipping creation"
fi

# Check if dataset already exists, push only if not
if ! flexai dataset list | grep -q "openhermes-fr"; then
  echo "Pushing dataset to FlexAI..."
  flexai dataset push "openhermes-fr" \
    --storage-provider "hf-conn-${RESOURCE_NAME}" \
    --source-path legmlai/openhermes-fr
else
  echo "Dataset 'openhermes-fr' already exists in FlexAI, skipping push"
fi


# Check if model already downloaded locally, download only if not
if [ ! -d "./Qwen2.5-7B" ]; then
  echo "Downloading model from HuggingFace..."
  python3 -m venv .venv_flexai_hf
  source .venv_flexai_hf/bin/activate
  pip install -r requirements.txt
  python ./download.py
  deactivate
else
  echo "Model 'Qwen2.5-7B' already downloaded locally, skipping download"
fi

: <<'COMMENT'
from huggingface_hub import snapshot_download

# Download Qwen2.5-7B model
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B",
    local_dir="./Qwen2.5-7B",
    repo_type="model"
)
print("Model downloaded successfully!")
COMMENT


# Check if checkpoint already exists in FlexAI, push only if not
if ! flexai checkpoint list | grep -q "hf-ckpt-${RESOURCE_NAME}"; then
  echo "Pushing checkpoint to FlexAI..."
  (
    cd ./Qwen2.5-7B && \
    rm -rf .cache && \
    rm -rf .gitattributes && \
    flexai checkpoint push "hf-ckpt-${RESOURCE_NAME}" --file ./
  )
else
  echo "Checkpoint 'hf-ckpt-${RESOURCE_NAME}' already exists in FlexAI, skipping push"
fi

# start the training run (still use unique UUID for run name to allow multiple runs)
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"

flexai training run "axolotl-french-sft-${RUN_UUID}" \
  --accels 4 --nodes 1 \ 
  #4 H400's of the cluster -- significant burden
  --repository-url https://github.com/kd2594/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN="hf-token-${RESOURCE_NAME}" \
  --checkpoint "hf-ckpt-${RESOURCE_NAME}" \
  --requirements-path code/axolotl/requirements.txt \
  -- axolotl train code/axolotl/qwen2/fft-7b-french.yaml

  # For linking private repo of user, user needs to install https://docs.flex.ai/cli/reference/code-registry 
  # For Docker image automation, instead of private repo, find workaround 
