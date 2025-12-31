#!/bin/bash
# Ensure this script is executable 
# chmod +x ./run-easyR1.sh

# Ensure all sibling .sh files are executable (useful after Drive/Git sync)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

# EasyR1 (showcased in flexaihq/blueprints/experiments/easyR1)
# Reference: https://github.com/flexaihq/blueprints/blob/main/experiments/easyR1/README.md

# Required secrets (paste values here or export them and replace assignments)
HF_TOKEN="paste-ur-token-here"
WANDB_API_KEY="paste-ur-wandb-key-here"

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is empty. Create a Hugging Face token and set it here."
  echo "Docs: https://huggingface.co/docs/hub/en/security-tokens"
  exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY is empty. Create a Weights & Biases API key and set it here."
  exit 1
fi

# Stable identifier for reusing FlexAI resources across runs
RESOURCE_NAME="$(whoami)-easyr1-qwen25"

HF_TOKEN_SECRET="hf-token-${RESOURCE_NAME}"
WANDB_SECRET="wandb-${RESOURCE_NAME}"
HF_CONN="hf-conn-${RESOURCE_NAME}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_CKPT_NAME="qwen25-7b-instruct-${RESOURCE_NAME}"

# Create secrets (idempotent)
if ! flexai secret list | grep -q "${HF_TOKEN_SECRET}"; then
  echo "Creating HF token secret ${HF_TOKEN_SECRET}..."
  echo "$HF_TOKEN" | flexai secret create "${HF_TOKEN_SECRET}"
else
  echo "HF token secret already exists, skipping creation"
fi

if ! flexai secret list | grep -q "${WANDB_SECRET}"; then
  echo "Creating W&B API key secret ${WANDB_SECRET}..."
  echo "$WANDB_API_KEY" | flexai secret create "${WANDB_SECRET}"
else
  echo "W&B API key secret already exists, skipping creation"
fi

# Optional: pre-fetch the base model to FlexAI as a checkpoint (recommended for faster start)
# This avoids downloading large weights at runtime.
if ! flexai storage list | grep -q "${HF_CONN}"; then
  echo "Creating HF storage connection ${HF_CONN}..."
  flexai storage create "${HF_CONN}" \
    --provider huggingface \
    --hf-token-name "${HF_TOKEN_SECRET}"
else
  echo "HF storage connection already exists, skipping creation"
fi

if ! flexai checkpoint list | grep -q "${MODEL_CKPT_NAME}"; then
  echo "Pre-fetching model as checkpoint ${MODEL_CKPT_NAME} (source: ${MODEL_ID})..."
  flexai checkpoint push "${MODEL_CKPT_NAME}" \
    --storage-provider "${HF_CONN}" \
    --source-path "${MODEL_ID}"
else
  echo "Model checkpoint already exists, skipping prefetch"
fi

# Start an EasyR1 GRPO training run
# Note: If you haven't connected FlexAI to GitHub, run: flexai code-registry connect
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"
RUN_NAME="easyr1-grpo-${RUN_UUID}"

flexai training run grpo \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path=Qwen/Qwen2.5-7B-Instruct

#  --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
#   --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \


: <<'COMMENT'
flexai training run "${RUN_NAME}" \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --checkpoint "${MODEL_CKPT_NAME}" \
  --env FORCE_TORCHRUN=1 \
  --env MODEL_CKPT_NAME="${MODEL_CKPT_NAME}" \
  --env HF_MODEL_ID="${MODEL_ID}" \
  --secret WANDB_API_KEY="${WANDB_SECRET}" \
  --secret HF_TOKEN="${HF_TOKEN_SECRET}" \
  --requirements-path code/easyR1/requirements.txt \
  --runtime pytorch-28-vllm-0110-nvidia \
  -- bash -lc '
    set -euo pipefail

    echo "Listing /input-checkpoint (if mounted):"
    ls -la /input-checkpoint || true

    # Transformers will treat a non-existent local path as a Hugging Face Hub repo id.
    # Pick a model path that actually exists; otherwise fall back to the Hub model id.
    if [ -d "/input-checkpoint/${MODEL_CKPT_NAME}" ]; then
      MODEL_PATH="/input-checkpoint/${MODEL_CKPT_NAME}"
    elif [ -f "/input-checkpoint/config.json" ] || [ -f "/input-checkpoint/tokenizer.json" ] || [ -f "/input-checkpoint/tokenizer_config.json" ]; then
      MODEL_PATH="/input-checkpoint"
    else
      MODEL_PATH="${HF_MODEL_ID}"
    fi

    echo "Using MODEL_PATH=${MODEL_PATH}"

    python3 -m verl.trainer.main \
      config=code/easyR1/config.yaml \
      worker.actor.model.model_path="${MODEL_PATH}" \
      worker.actor.model.tokenizer_path="${MODEL_PATH}"
  '

# Useful commands:
# flexai training inspect "${RUN_NAME}"
# flexai training logs "${RUN_NAME}"
# flexai training checkpoints "${RUN_NAME}"

COMMENT
