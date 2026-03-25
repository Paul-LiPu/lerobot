#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash infer.sh
#   bash infer.sh my-eval-dataset
#   bash infer.sh my-eval-dataset "pick up the plate using suction cup"
#   bash infer.sh my-eval-dataset "pick up the plate using suction cup" my-policy
#   bash infer.sh my-eval-dataset "pick up the plate using suction cup" my-policy /path/to/initial_pose.json
#
# Notes:
#   - Arg 1: eval dataset name or full repo id
#   - Arg 2: task string
#   - Arg 3: policy name or full repo id
#   - Arg 4: initial pose path
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_EVAL_DATASET_NAME="eval_lebai_act_romoya_grip_black_taped_box"
# DEFAULT_SINGLE_TASK="pick up the plate using suction cup"
DEFAULT_SINGLE_TASK="Grab the ingredient box with gripper"
# DEFAULT_POLICY_NAME="act_sjadj_lebai-suction-plate"
# DEFAULT_POLICY_NAME="act_sjadj_lebai-suction-plate_side"
# DEFAULT_POLICY_NAME="act_sjaj_lebai-suction-plate_side"
# DEFAULT_POLICY_NAME="act_sjaj_lebai-gripper-box"
DEFAULT_POLICY_NAME="act_sjaj_lebai-gripper-black-tape-box"
DEFAULT_N_ACTION_STEPS=30
DEFAULT_NUM_EPISODES=10
DEFAULT_EPISODE_TIME_S=60
DEFAULT_RESET_TIME_S=60
DEFAULT_DEBUG=0
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-360p}"

case "${CAMERA_RESOLUTION}" in
  1080p)
    CAMERA_WIDTH=1920
    CAMERA_HEIGHT=1080
    ;;
  720p)
    CAMERA_WIDTH=1280
    CAMERA_HEIGHT=720
    ;;
  360p)
    CAMERA_WIDTH=640
    CAMERA_HEIGHT=360
    ;;
  *)
    echo "Unsupported CAMERA_RESOLUTION '${CAMERA_RESOLUTION}'. Use one of: 1080p, 720p, 360p." >&2
    exit 1
    ;;
esac

EVAL_DATASET_NAME="${1:-${DEFAULT_EVAL_DATASET_NAME}}"
SINGLE_TASK="${2:-${DEFAULT_SINGLE_TASK}}"
POLICY_NAME="${3:-${DEFAULT_POLICY_NAME}}"
INITIAL_POSE_PATH_ARG="${4:-}"
# DEFAULT_INITIAL_POSE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/initial_pose_lebai-suction-plate_25.json"
DEFAULT_INITIAL_POSE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/initial_pose_lebai-gripper-box_0.json"
INITIAL_POSE_PATH="${INITIAL_POSE_PATH_ARG:-${INITIAL_POSE_PATH:-${DEFAULT_INITIAL_POSE_PATH}}}"

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

EXTRA_ARGS=()
if [[ -n "${INITIAL_POSE_PATH}" ]]; then
  EXTRA_ARGS+=(--initial_pose_path="${INITIAL_POSE_PATH}")
fi

bash setup_cams.sh "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}"

  # --resume=true \
  #   --robot.gripper_closed_position=82.0 \
LEROBOT_DEBUG="${DEFAULT_DEBUG}" uv run --extra romoya lerobot-record \
  --dataset.fps=30 \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
  --robot.trace_path=infer-trace.json \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: /dev/cam_top, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: /dev/cam_side, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}}" \
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
  --policy.path="${POLICY_PATH}" \
  --policy.n_action_steps="${DEFAULT_N_ACTION_STEPS}" \
  "${EXTRA_ARGS[@]}"
