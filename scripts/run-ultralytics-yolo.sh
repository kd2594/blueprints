#!/bin/bash

# Ensure this script is executable 
# chmod +x ./run-ultralytics-yolo.sh

# Install FlexAI CLI if not already installed
if command -v flexai >/dev/null 2>&1; then
  echo "FlexAI CLI is already installed"
else
  echo "Installing FlexAI CLI..."
  curl -fsSL https://cli.flex.ai/install.sh | sh
fi

# Ensure FlexAI CLI is in PATH for current session and future sessions
export PATH="/home/flexai/.flexai/bin:$PATH" && echo "Current shell: $SHELL" && command -v flexai && flexai --version && for f in "$HOME/.bashrc" "$HOME/.profile"; do [ -f "$f" ] || continue; grep -q '/home/flexai/.flexai/bin' "$f" || printf '\n# FlexAI CLI\nexport PATH="/home/flexai/.flexai/bin:$PATH"\n' >> "$f"; done && echo "Persisted in: $(grep -l '/home/flexai/.flexai/bin' $HOME/.bashrc $HOME/.profile 2>/dev/null | tr '\n' ' ')"

# Use a fixed identifier instead of random UUID to enable reuse
# Change this to a unique name for different projects
RESOURCE_NAME="$(whoami)-yolo11-detection"

# Configuration: Set to "coco8" for quick testing or "custom" for your own dataset
DATASET_MODE="coco8"  # Options: "coco8" or "custom"
CUSTOM_DATASET_NAME="yolo-custom-dataset"  # Only used if DATASET_MODE="custom"

# Model configuration
YOLO_MODEL="yolo11n"  # Options: yolo11n, yolo11s, yolo11m, yolo11l
TASK_TYPE="detect"    # Options: detect, segment, pose

# Training hyperparameters
NUM_ACCELS=4          # Number of GPUs
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640

# Optional: Set up custom dataset if DATASET_MODE="custom"
if [ "$DATASET_MODE" = "custom" ]; then
  # Check if dataset already exists, push only if not
  if ! flexai dataset list | grep -q "$CUSTOM_DATASET_NAME"; then
    echo "Pushing custom dataset to FlexAI..."
    echo "Make sure your dataset is in YOLO format with data.yaml at the root"
    # flexai dataset push "$CUSTOM_DATASET_NAME" --file path/to/your-dataset
    echo "⚠️  Please uncomment and update the line above with your dataset path"
    exit 1
  else
    echo "Dataset '$CUSTOM_DATASET_NAME' already exists in FlexAI"
  fi
fi

# Generate unique run name
RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"
RUN_NAME="yolo11-${TASK_TYPE}-${RUN_UUID}"

echo "Starting YOLO11 training job: ${RUN_NAME}"
echo "Model: ${YOLO_MODEL}"
echo "Task: ${TASK_TYPE}"
echo "Dataset: ${DATASET_MODE}"
echo "GPUs: ${NUM_ACCELS}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"

# Build the training command based on dataset mode
if [ "$DATASET_MODE" = "coco8" ]; then
  # COCO8 dataset will be auto-downloaded by Ultralytics
  DATA_CONFIG="coco8.yaml"
  DATASET_FLAG=""
  
  flexai training run "${RUN_NAME}" \
    --accels ${NUM_ACCELS} --nodes 1 \
    --repository-url https://github.com/flexaihq/blueprints \
    --requirements-path code/ultralytics/requirements.txt \
    -- yolo ${TASK_TYPE} train \
      data=${DATA_CONFIG} \
      model=${YOLO_MODEL}.pt \
      epochs=${EPOCHS} \
      batch=${BATCH_SIZE} \
      imgsz=${IMAGE_SIZE} \
      project=/output \
      name=train \
      patience=50 \
      save=True \
      device=0,1,2,3 \
      workers=8 \
      val=True

else
  # Custom dataset mode - mount dataset to /input
  DATA_CONFIG="/input/data.yaml"
  DATASET_FLAG="--dataset ${CUSTOM_DATASET_NAME}"
  
  flexai training run "${RUN_NAME}" \
    --accels ${NUM_ACCELS} --nodes 1 \
    --repository-url https://github.com/flexaihq/blueprints \
    ${DATASET_FLAG} \
    --requirements-path code/ultralytics/requirements.txt \
    -- yolo ${TASK_TYPE} train \
      data=${DATA_CONFIG} \
      model=${YOLO_MODEL}.pt \
      epochs=${EPOCHS} \
      batch=${BATCH_SIZE} \
      imgsz=${IMAGE_SIZE} \
      project=/output \
      name=train \
      patience=50 \
      save=True \
      device=0,1,2,3 \
      workers=8 \
      val=True
fi

# Check if the training command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Training job submitted successfully!"
    echo "Run name: ${RUN_NAME}"
    echo ""
    echo "Useful commands:"
    echo "  flexai training inspect ${RUN_NAME}"
    echo "  flexai training logs ${RUN_NAME}"
    echo "  flexai training checkpoints ${RUN_NAME}"
    echo "  flexai training debug-ssh ${RUN_NAME}"
else
    echo "Error: Training job submission failed"
    exit 1
fi

: <<'COMMENT'

=== YOLO11 Training Script for FlexAI ===

This script trains YOLO11 models for computer vision tasks on FlexAI.

CONFIGURATION OPTIONS:
- DATASET_MODE: "coco8" (auto-download) or "custom" (use your dataset)
- YOLO_MODEL: yolo11n (nano), yolo11s (small), yolo11m (medium), yolo11l (large)
- TASK_TYPE: detect (object detection), segment (instance segmentation), pose (pose estimation)
- NUM_ACCELS: Number of GPUs to use (1, 4, 8)
- EPOCHS: Number of training epochs (100 recommended for COCO8)
- BATCH_SIZE: Batch size per GPU (adjust based on model size and GPU memory)
- IMAGE_SIZE: Input image size (640 standard, 1280 for small objects)

DATASET FORMAT (Custom datasets):
Your dataset should follow YOLO format:
  dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/

And include a data.yaml file at the root:
  path: /input
  train: /input/images/train
  val: /input/images/val
  names:
    0: person
    1: bicycle
    2: car

EXPECTED RESULTS (YOLO11n on COCO8):
- mAP50: 0.45-0.55
- mAP50-95: 0.30-0.40
- Training time: 5-10 minutes (4 × H100, 100 epochs)
- Model size: ~3 MB

CHECKPOINTS:
Training outputs are saved to /output including:
- weights/best.pt (best model)
- weights/last.pt (last epoch)
- Training metrics and curves

To download checkpoints:
  flexai training checkpoints ${RUN_NAME}
  flexai checkpoint fetch "<CHECKPOINT_ID>" --destination ./yolo11-checkpoint

VALIDATION:
After training, validate your model:
  flexai training run yolo11-validation \
    --accels 1 --nodes 1 \
    --repository-url https://github.com/flexaihq/blueprints \
    --checkpoint "<CHECKPOINT_ID>" \
    --requirements-path code/ultralytics/requirements.txt \
    -- yolo detect val \
      data=coco8.yaml \
      model=/checkpoint/weights/best.pt \
      imgsz=640

INFERENCE:
Run predictions on new images:
  flexai training run yolo11-predict \
    --accels 1 --nodes 1 \
    --repository-url https://github.com/flexaihq/blueprints \
    --checkpoint "<CHECKPOINT_ID>" \
    --dataset test-images \
    --requirements-path code/ultralytics/requirements.txt \
    -- yolo detect predict \
      model=/checkpoint/weights/best.pt \
      source=/input \
      imgsz=640 \
      conf=0.25 \
      save=True \
      project=/output

TROUBLESHOOTING:
- Out of memory: Reduce BATCH_SIZE or IMAGE_SIZE
- Low mAP: Increase EPOCHS, use larger model, or increase IMAGE_SIZE
- Dataset errors: Verify YOLO format and data.yaml paths
- Training too slow: Increase NUM_ACCELS or decrease BATCH_SIZE

For more information, see:
- Ultralytics YOLO Docs: https://docs.ultralytics.com/
- FlexAI Docs: https://docs.flex.ai/

COMMENT
