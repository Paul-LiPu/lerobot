#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash extract_initial_pose.sh
#   bash extract_initial_pose.sh my-dataset
#   bash extract_initial_pose.sh my-dataset 0
#   bash extract_initial_pose.sh my-dataset 0 /path/to/initial_pose.json

DEFAULT_DATASET_NAME="record-test"
DEFAULT_EPISODE=0

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
EPISODE="${2:-${DEFAULT_EPISODE}}"
DATASET_BASENAME="${DATASET_NAME##*/}"
SAFE_DATASET_BASENAME="${DATASET_BASENAME//[^A-Za-z0-9._-]/_}"
OUTPUT_PATH="${3:-./initial_pose_${SAFE_DATASET_BASENAME}_${EPISODE}.json}"

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

uv run --extra romoya python extract_initial_pose.py \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --episode "${EPISODE}" \
  --output "${OUTPUT_PATH}"
