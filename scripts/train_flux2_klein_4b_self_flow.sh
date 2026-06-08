#!/usr/bin/env bash
set -euo pipefail

cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

CONFIG="configs/train/flux2_klein_4b_self_flow.yaml"
ACCELERATE_CONFIG="configs/train/accelerate_flux2_self_flow_zero3.yaml"
TRAIN_SCRIPT="examples/flux2/model_training/train_self_flow.py"
BASE_MODEL="/mnt/image-edit/datasets/dingbaojin/models/black-forest-labs/FLUX.2-klein-4B"
OUTPUT_DIR="/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_4b_self_flow"

has_deepspeed() {
  python -c "import deepspeed" >/dev/null 2>&1
}

MODE="${1:-train}"
if [[ "${MODE}" == "smoke" || "${MODE}" == "--smoke" ]]; then
  shift || true
  if has_deepspeed; then
    echo "Running the 8-GPU DeepSpeed dummy-data smoke test for 2 optimizer steps."
    LAUNCH_CMD=(accelerate launch --config_file "${ACCELERATE_CONFIG}")
  else
    SMOKE_NUM_PROCESSES="${SMOKE_NUM_PROCESSES:-1}"
    echo "DeepSpeed is not installed; running smoke test without DeepSpeed on ${SMOKE_NUM_PROCESSES} process(es)."
    echo "For full 4B training, install DeepSpeed in this environment first."
    LAUNCH_CMD=(accelerate launch --num_processes "${SMOKE_NUM_PROCESSES}" --mixed_precision bf16)
  fi
  "${LAUNCH_CMD[@]}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}/smoke" \
    --dataset_type dummy \
    --height 128 \
    --width 128 \
    --max_steps 2 \
    --checkpointing_steps 2 \
    --gradient_accumulation_steps 1 \
    --use_gradient_checkpointing \
    "$@"
else
  if ! has_deepspeed; then
    echo "DeepSpeed is required for the default 8-GPU full-parameter training path."
    echo "Install it in the active environment, then rerun this script."
    exit 1
  fi
  echo "Running 8-GPU FLUX.2-klein-4B full-parameter Self-Flow training."
  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_gradient_checkpointing \
    "$@"
fi
