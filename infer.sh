#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash infer.sh
#   bash infer.sh my-eval-dataset
#   bash infer.sh my-eval-dataset "pick up the plate using suction cup"
#   bash infer.sh my-eval-dataset "pick up the plate using suction cup" my-policy
#
# Notes:
#   - Arg 1: eval dataset name or full repo id
#   - Arg 2: task string
#   - Arg 3: policy name or full repo id
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_EVAL_DATASET_NAME="eval_lebai_act_romoya_sunction_plate"
DEFAULT_SINGLE_TASK="pick up the plate using suction cup"
DEFAULT_POLICY_NAME="act_sjadj_lebai-suction-plate"
DEFAULT_NUM_EPISODES=10
DEFAULT_EPISODE_TIME_S=60
DEFAULT_RESET_TIME_S=60

EVAL_DATASET_NAME="${1:-${DEFAULT_EVAL_DATASET_NAME}}"
SINGLE_TASK="${2:-${DEFAULT_SINGLE_TASK}}"
POLICY_NAME="${3:-${DEFAULT_POLICY_NAME}}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

if [[ "${EVAL_DATASET_NAME}" == */* ]]; then
  DATASET_REPO_ID="${EVAL_DATASET_NAME}"
else
  DATASET_REPO_ID="${HF_USER}/${EVAL_DATASET_NAME}"
fi

if [[ "${POLICY_NAME}" == */* ]]; then
  POLICY_PATH="${POLICY_NAME}"
else
  POLICY_PATH="${HF_USER}/${POLICY_NAME}"
fi

uv run --extra romoya lerobot-record \
  --dataset.fps=30 \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: 640, height: 360, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: /dev/cam_top, width: 640, height: 360, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: /dev/cam_side, width: 640, height: 360, fps: 30, fourcc: MJPG}}' \
  --robot.gripper_closed_position=82.0 \
  --robot.gripper_force=100 \
  --robot.acceleration=1.0 \
  --robot.velocity=1.0 \
  --robot.blend_radius=0.0 \
  --teleop.type=romoya_lebai_leader \
  --teleop.ip=192.168.50.154 \
  --teleop.gripper_force=100 \
  --teleop.id=lebai_leader_arm \
  --display_data=false \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.num_episodes="${DEFAULT_NUM_EPISODES}" \
  --dataset.episode_time_s="${DEFAULT_EPISODE_TIME_S}" \
  --dataset.reset_time_s="${DEFAULT_RESET_TIME_S}" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=4 \
  --policy.path="${POLICY_PATH}"
