#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-text2video.sh

set -euo pipefail

# Text-to-Video Inference (Wan2.2) on FlexAI
# - Creates/uses a HF token secret
# - Creates/uses an inference endpoint
# - Extracts endpoint URL
# - Runs a sample generation curl (writes an .mp4)

# -------- Config (edit as needed) --------
RESOURCE_NAME="$(whoami)-wan-t2v"

INFERENCE_NAME="${INFERENCE_NAME:-wan-text-to-video-${RESOURCE_NAME}}"

HF_SECRET_NAME="${HF_SECRET_NAME:-MY_HF_TOKEN-${RESOURCE_NAME}}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"

TASK="${TASK:-text-to-video}"
QUANT="${QUANT:-bitsandbytes_4bit}"

OUT_FILE="${OUT_FILE:-boxing_cats.mp4}"
PROMPT_TEXT="${PROMPT_TEXT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"

NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
STEPS="${STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-3.0}"
# ----------------------------------------

# Ensure HF_TOKEN is available (you can also `export HF_TOKEN=...` before running)
# HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN="paste-ur-token-here"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Error: HF_TOKEN is not set. Export it first (must have access to ${MODEL_ID})."
  echo "Example: export HF_TOKEN=hf_xxx"
  exit 1
fi

# Create secret if missing
if ! flexai secret list | grep -q "${HF_SECRET_NAME}"; then
  echo "Creating FlexAI secret: ${HF_SECRET_NAME}"
  printf "%s" "${HF_TOKEN}" | flexai secret create "${HF_SECRET_NAME}"
else
  echo "Secret '${HF_SECRET_NAME}' already exists, skipping"
fi

# Create endpoint if missing
if flexai inference list 2>/dev/null | grep -q "${INFERENCE_NAME}"; then
  echo "Inference '${INFERENCE_NAME}' already exists, skipping serve"
else
  echo "Starting inference endpoint: ${INFERENCE_NAME}"
  flexai inference serve "${INFERENCE_NAME}" \
    --runtime flexserve \
    --hf-token-secret "${HF_SECRET_NAME}" \
    -- \
    --task "${TASK}" \
    --model "${MODEL_ID}" \
    --quantization-config "${QUANT}"
fi

# Get endpoint URL (jq preferred; fall back to python)
echo "Fetching endpoint URL..."
INSPECT_JSON="$(flexai inference inspect "${INFERENCE_NAME}" -j)"

if command -v jq >/dev/null 2>&1; then
  INFERENCE_URL="$(printf '%s' "${INSPECT_JSON}" | jq -r '.config.endpointUrl')"
else
  INFERENCE_URL="$(
    python3 - <<'PY'
import json,sys
obj=json.load(sys.stdin)
print(obj["config"]["endpointUrl"])
PY
  <<< "${INSPECT_JSON}"
  )"
fi

if [[ -z "${INFERENCE_URL}" || "${INFERENCE_URL}" == "null" ]]; then
  echo "Error: could not determine endpointUrl from: flexai inference inspect ${INFERENCE_NAME} -j"
  exit 1
fi

# API key: must come from endpoint creation output (paste once) or be exported
INFERENCE_API_KEY="${INFERENCE_API_KEY:-}"
if [[ -z "${INFERENCE_API_KEY}" ]]; then
  echo "Paste the INFERENCE API key shown when the endpoint was created."
  read -r -s -p "INFERENCE_API_KEY: " INFERENCE_API_KEY
  echo
fi

if [[ -z "${INFERENCE_API_KEY}" ]]; then
  echo "Error: INFERENCE_API_KEY is empty."
  exit 1
fi

echo "Endpoint URL: ${INFERENCE_URL}"
echo "Generating video -> ${OUT_FILE}"

curl -sS -X POST \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"inputs\": \"${PROMPT_TEXT}\",
    \"parameters\": {
      \"num_frames\": ${NUM_FRAMES},
      \"height\": ${HEIGHT},
      \"width\": ${WIDTH},
      \"num_inference_steps\": ${STEPS},
      \"guidance_scale\": ${GUIDANCE_SCALE},
      \"guidance_scale_2\": ${GUIDANCE_SCALE_2}
    }
  }" \
  -o "${OUT_FILE}" \
  "${INFERENCE_URL}/v1/videos/generations"

echo "Done: ${OUT_FILE}"
