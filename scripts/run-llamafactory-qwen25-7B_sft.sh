#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Ensure this script is executable 
# chmod +x ./run-llamafactory.sh

# Install FlexAI CLI if not already installed
if command -v flexai >/dev/null 2>&1; then
  echo "FlexAI CLI is already installed"
else
  echo "Installing FlexAI CLI..."
  curl -fsSL https://cli.flex.ai/install.sh | sh
fi

# Ensure FlexAI CLI is in PATH for current session (macOS-safe)
export PATH="$HOME/.flexai/bin:$PATH"
echo "Current shell: $SHELL"
command -v flexai

# Provide HF token via env var (recommended): export HF_TOKEN=...
HF_TOKEN="${HF_TOKEN:-}"

# Prompt for token if not provided
if [ -z "$HF_TOKEN" ]; then
  if [ -t 0 ]; then
    # Interactive terminal: prompt with hidden input
    read -r -s -p "HF_TOKEN (input hidden): " HF_TOKEN
    echo
  else
    # Non-interactive stdin: still try reading from the controlling TTY
    read -r -s -p "HF_TOKEN (input hidden): " HF_TOKEN </dev/tty
    echo >/dev/tty
  fi
fi

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is empty. Please provide a Hugging Face token. https://huggingface.co/docs/hub/en/security-tokens"
  exit 1
fi

# Use a fixed identifier instead of random UUID to enable reuse
# Change this to a unique name for different projects
RESOURCE_NAME="$(whoami)-qwen25-instruct-openhermes"


# Check if secret already exists, create only if not
if flexai secret inspect "hf-token-${RESOURCE_NAME}" >/dev/null 2>&1; then
  echo "HF token secret already exists, skipping creation"
else
  echo "Creating HF token secret..."
  echo "$HF_TOKEN" | flexai secret create "hf-token-${RESOURCE_NAME}"
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
if flexai dataset inspect "openhermes-fr" >/dev/null 2>&1; then
  echo "Dataset 'openhermes-fr' already exists in FlexAI, skipping push"
else
  echo "Pushing dataset to FlexAI..."
  flexai dataset push "openhermes-fr" \
    --storage-provider "hf-conn-${RESOURCE_NAME}" \
    --source-path legmlai/openhermes-fr
fi


# Check if checkpoint already exists in FlexAI, push only if not
if ! flexai checkpoint list | grep -q "hf-ckpt-${RESOURCE_NAME}"; then
  echo "Pushing checkpoint to FlexAI..."
  # Prefer pre-fetching directly from Hugging Face via the FlexAI storage connection.
  # This avoids local path issues (missing requirements.txt/download.py) when you run from other folders.
  flexai checkpoint push "hf-ckpt-${RESOURCE_NAME}" \
    --storage-provider "hf-conn-${RESOURCE_NAME}" \
    --source-path Qwen/Qwen2.5-7B
else
  echo "Checkpoint 'hf-ckpt-${RESOURCE_NAME}' already exists in FlexAI, skipping push"
fi

# start the training run (still use unique UUID for run name to allow multiple runs)
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"

RUN_NAME="qwen25-7b-all-${RUN_UUID}"

# Use an array to avoid line-continuation issues that can cause:
# "Invalid number of arguments: requires at least 2 arg(s), only received 1"
TRAIN_CMD=(
  flexai training run "$RUN_NAME"
  --accels 8 --nodes 1
  --repository-url https://github.com/flexaihq/blueprints
  --env FORCE_TORCHRUN=1
  --secret "HF_TOKEN=hf-token-${RESOURCE_NAME}"
  --checkpoint "hf-ckpt-${RESOURCE_NAME}"
  --requirements-path code/llama-factory/requirements.txt
  --dataset openhermes-fr
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml
)

"${TRAIN_CMD[@]}"
  
  #--dataset openhermes-fr 
  #--checkpoint "hf-ckpt-${RESOURCE_NAME}" \

  #-- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7b-instruct_sft_lora.yml
  
  #-- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-prefetched_all_sft.yaml save_strategy=steps save_steps=50000

  # yet to test
  #-- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml 


