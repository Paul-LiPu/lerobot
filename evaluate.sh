#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash evaluate.sh
#   bash evaluate.sh lebai-suction-plate
#   bash evaluate.sh lebai-suction-plate act_sjadj_lebai-suction-plate
#   bash evaluate.sh lebai-suction-plate act_sjadj_lebai-suction-plate cpu
#
# Notes:
#   - Arg 1: dataset name or full repo id
#   - Arg 2: policy name or full repo id
#   - Arg 3: device
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_DATASET_NAME="lebai-suction-plate"
DEFAULT_POLICY_NAME="act_sjadj_lebai-suction-plate"
DEFAULT_DEVICE="cuda"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
POLICY_NAME="${2:-${DEFAULT_POLICY_NAME}}"
DEVICE="${3:-${DEFAULT_DEVICE}}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

if [[ "${DATASET_NAME}" == */* ]]; then
  DATASET_REPO_ID="${DATASET_NAME}"
else
  DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"
fi

if [[ "${POLICY_NAME}" == */* ]]; then
  POLICY_PATH="${POLICY_NAME}"
else
  POLICY_PATH="${HF_USER}/${POLICY_NAME}"
fi

uv run --extra romoya python eval_policy_on_dataset.py \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --policy-path "${POLICY_PATH}" \
  --device "${DEVICE}"
