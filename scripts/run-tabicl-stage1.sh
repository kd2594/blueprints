#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-tabicl-stage1.sh

# Install FlexAI CLI if not already installed
if command -v flexai >/dev/null 2>&1; then
  echo "FlexAI CLI is already installed"
else
  echo "Installing FlexAI CLI..."
  curl -fsSL https://cli.flex.ai/install.sh | sh
fi

# Ensure FlexAI CLI is in PATH for current session and future sessions
export PATH="/home/flexai/.flexai/bin:$PATH" && echo "Current shell: $SHELL" && command -v flexai && flexai --version && for f in "$HOME/.bashrc" "$HOME/.profile"; do [ -f "$f" ] || continue; grep -q '/home/flexai/.flexai/bin' "$f" || printf '\n# FlexAI CLI\nexport PATH="/home/flexai/.flexai/bin:$PATH"\n' >> "$f"; done && echo "Persisted in: $(grep -l '/home/flexai/.flexai/bin' $HOME/.bashrc $HOME/.profile 2>/dev/null | tr '\n' ' ')"

# FlexAI training requires a requirements file to exist in the repository referenced by
# --repository-url. The TabICL repo only ships a pyproject.toml, so we use the FlexAI
# blueprints repo as the build context, and clone TabICL inside the job.
BLUEPRINTS_REPOSITORY_URL="https://github.com/flexaihq/blueprints"
BLUEPRINTS_REQUIREMENTS_PATH="code/causal-language-modeling/requirements.txt"

# TabICL repo containing the training code
# IMPORTANT: do NOT wrap the URL in angle brackets (<...>) in zsh/bash.
TABICL_REPOSITORY_URL="https://github.com/soda-inria/tabicl.git"

# Run naming
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"
RUN_NAME="tabicl-stage1-${RUN_UUID}"

# Keep these aligned with your interactive debug-ssh command
WANDB_PROJECT="TabICL"
WANDB_NAME="Stage1"
WANDB_DIR="/output"
CHECKPOINT_DIR="/output"

flexai training run "${RUN_NAME}" \
  --accels 1 --nodes 1 \
  --repository-url "${BLUEPRINTS_REPOSITORY_URL}" \
  --requirements-path "${BLUEPRINTS_REQUIREMENTS_PATH}" \
  --env FORCE_TORCHRUN=1 \
  -- bash -lc "
set -euo pipefail

# Match your interactive setup
git clone ${TABICL_REPOSITORY_URL} tabicl
cd tabicl
pip install -e .
cd src/tabicl

torchrun --standalone --nproc_per_node=1 train/run.py \\
  --wandb_log False \\
  --wandb_project ${WANDB_PROJECT} \\
  --wandb_name ${WANDB_NAME} \\
  --wandb_dir ${WANDB_DIR} \\
  --wandb_mode offline \\
  --device cuda \\
  --dtype float32 \\
  --np_seed 42 \\
  --torch_seed 42 \\
  --max_steps 100000 \\
  --batch_size 512 \\
  --micro_batch_size 4 \\
  --lr 1e-4 \\
  --scheduler cosine_warmup \\
  --warmup_proportion 0.02 \\
  --gradient_clipping 1.0 \\
  --prior_type mix_scm \\
  --prior_device cpu \\
  --batch_size_per_gp 4 \\
  --min_features 2 \\
  --max_features 100 \\
  --max_classes 10 \\
  --max_seq_len 1024 \\
  --min_train_size 0.1 \\
  --max_train_size 0.9 \\
  --embed_dim 128 \\
  --col_num_blocks 3 \\
  --col_nhead 4 \\
  --col_num_inds 128 \\
  --row_num_blocks 3 \\
  --row_nhead 8 \\
  --row_num_cls 4 \\
  --row_rope_base 100000 \\
  --icl_num_blocks 12 \\
  --icl_nhead 4 \\
  --ff_factor 2 \\
  --norm_first True \\
  --checkpoint_dir ${CHECKPOINT_DIR} \\
  --save_temp_every 50 \\
  --save_perm_every 5000
"

flexai training inspect "${RUN_NAME}"

# flexai training logs "${RUN_NAME}"
# flexai training debug-ssh "${RUN_NAME}"
