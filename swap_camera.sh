#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash swap_camera.sh
#   bash swap_camera.sh suction-plate wrist top
#   bash swap_camera.sh suction-plate wrist top true
#
# Notes:
#   - Arg 1: dataset name
#   - Arg 2: first camera name
#   - Arg 3: second camera name
#   - Arg 4: whether to overwrite the original dataset (`true` or `false`)
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_DATASET_NAME="suction-plate"
DEFAULT_CAMERA_A="wrist"
DEFAULT_CAMERA_B="top"
DEFAULT_OUTPUT_SUFFIX="-fixed"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
CAMERA_A="${2:-${DEFAULT_CAMERA_A}}"
CAMERA_B="${3:-${DEFAULT_CAMERA_B}}"
OVERWRITE_ORIGINAL="${4:-false}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

if [[ "${OVERWRITE_ORIGINAL}" == "true" ]]; then
  OUTPUT_REPO_ID="${DATASET_REPO_ID}"
else
  OUTPUT_REPO_ID="${HF_USER}/${DATASET_NAME}${DEFAULT_OUTPUT_SUFFIX}"
fi

uv run lerobot-swap-camera-names \
  --repo_id="${DATASET_REPO_ID}" \
  --camera_a="${CAMERA_A}" \
  --camera_b="${CAMERA_B}" \
  --new_repo_id="${OUTPUT_REPO_ID}" \
  --push_to_hub=true
