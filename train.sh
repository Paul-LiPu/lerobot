#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train.sh
#   bash train.sh my-dataset
#   bash train.sh my-dataset my-policy
#   bash train.sh my-dataset my-policy my-run-name
#
# Notes:
#   - Arg 1: dataset name or full repo id
#   - Arg 2: policy repo name or full repo id
#   - Arg 3: local output/job name
#   - HF_USER is detected automatically from `hf auth whoami`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_act_romoya_config.json}"

DEFAULT_DATASET_NAME="lebai-suction-plate"
DEFAULT_POLICY_REPO_NAME="act_sjadj_lebai-suction-plate"
DEFAULT_OUTPUT_NAME="act_sjadj_lebai-suction-plate"
DEFAULT_POLICY_TYPE="act_romoya"
DEFAULT_DEVICE="cuda"
DEFAULT_STEPS=100000
DEFAULT_BATCH_SIZE=56
DEFAULT_WANDB_ENABLE="true"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
POLICY_REPO_NAME="${2:-${DEFAULT_POLICY_REPO_NAME}}"
OUTPUT_NAME="${3:-${DEFAULT_OUTPUT_NAME}}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

if [[ "${DEFAULT_WANDB_ENABLE}" == "true" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    uv run wandb login --relogin "${WANDB_API_KEY}"
  elif [[ ! -f "${HOME}/.netrc" ]] || ! grep -q 'machine api\.wandb\.ai' "${HOME}/.netrc"; then
    echo "wandb is not logged in. Starting wandb login..."
    uv run wandb login
  fi
fi

if [[ "${DATASET_NAME}" == */* ]]; then
  DATASET_REPO_ID="${DATASET_NAME}"
else
  DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"
fi

if [[ "${POLICY_REPO_NAME}" == */* ]]; then
  POLICY_REPO_ID="${POLICY_REPO_NAME}"
else
  POLICY_REPO_ID="${HF_USER}/${POLICY_REPO_NAME}"
fi

OUTPUT_DIR="outputs/train/${OUTPUT_NAME}"
JOB_NAME="${OUTPUT_NAME}"

uv run --extra romoya lerobot-train \
  --config_path="${CONFIG_PATH}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type="${DEFAULT_POLICY_TYPE}" \
  --policy.device="${DEFAULT_DEVICE}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${DEFAULT_STEPS}" \
  --batch_size="${DEFAULT_BATCH_SIZE}" \
  --wandb.enable="${DEFAULT_WANDB_ENABLE}"
