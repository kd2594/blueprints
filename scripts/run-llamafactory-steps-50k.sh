#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-llamafactory.sh

# Install FlexAI CLI if not already installed
if command -v flexai >/dev/null 2>&1; then
  echo "FlexAI CLI is already installed"
else
  echo "Installing FlexAI CLI..."
  curl -fsSL https://cli.flex.ai/install.sh | sh
fi

# Ensure FlexAI CLI is in PATH for current session and future sessions
export PATH="/home/flexai/.flexai/bin:$PATH" && echo "Current shell: $SHELL" && command -v flexai && flexai --version && for f in "$HOME/.bashrc" "$HOME/.profile"; do [ -f "$f" ] || continue; grep -q '/home/flexai/.flexai/bin' "$f" || printf '\n# FlexAI CLI\nexport PATH="/home/flexai/.flexai/bin:$PATH"\n' >> "$f"; done && echo "Persisted in: $(grep -l '/home/flexai/.flexai/bin' $HOME/.bashrc $HOME/.profile 2>/dev/null | tr '\n' ' ')"

HF_TOKEN="paste-ur-token-here"

# Check if HF_TOKEN is set and not empty
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please export your Hugging Face token as HF_TOKEN. https://huggingface.co/docs/hub/en/security-tokens"
  exit 1
fi

# Use a fixed identifier instead of random UUID to enable reuse
# Change this to a unique name for different projects
RESOURCE_NAME="$(whoami)-qwen25-openhermes"


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
flexai training run "qwen25-7b-all-${RUN_UUID}"  \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/flexaihq/blueprints \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN="hf-token-${RESOURCE_NAME}" \
  --checkpoint "hf-ckpt-${RESOURCE_NAME}" \
  --requirements-path code/llama-factory/requirements.txt \
  --dataset "openhermes-fr" \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-prefetched_all_sft.yaml save_strategy=steps save_steps=50000
  #-- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml 

: <<'COMMENT'

Current script from blueprint https://github.com/flexaihq/blueprints/blob/main/experiments/llama-factory/README.md is failing with checkpoint size disk space issue 

With save_steps: 50000, fune-tuning succeeds as it controls checkpoint saving frequency by saving a checkpoint every 50,000 training steps. 
Useful for long runs to prevent losing progress and manage disk space
Related to training duration/iterations, not dataset size

COMMENT

