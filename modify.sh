#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATASET_NAME="record-test"
DEFAULT_EPISODE_INDICES='[16, 23]'

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
EPISODE_INDICES="${2:-${DEFAULT_EPISODE_INDICES}}"

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

uv run lerobot-edit-dataset \
  --repo_id="${DATASET_REPO_ID}" \
  --operation.type=delete_episodes \
  --operation.episode_indices="${EPISODE_INDICES}" \
  --push_to_hub=true
