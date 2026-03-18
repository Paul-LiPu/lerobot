#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash replay.sh
#   bash replay.sh my-dataset
#   bash replay.sh my-dataset 0
#   bash replay.sh PL2011/my-dataset 3
#
# Notes:
#   - Arg 1: dataset name or full repo id
#   - Arg 2: episode index
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_DATASET_NAME="record-test"
DEFAULT_EPISODE="0"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
EPISODE="${2:-${DEFAULT_EPISODE}}"

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

uv run --extra romoya lerobot-replay \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
  --robot.gripper_closed_position=82.0 \
  --robot.gripper_force=100 \
  --robot.acceleration=1.0 \
  --robot.velocity=1.0 \
  --robot.blend_radius=0.0 \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.episode="${EPISODE}"
