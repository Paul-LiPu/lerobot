#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash infer_pi05.sh
#   bash infer_pi05.sh my-eval-dataset
#   bash infer_pi05.sh my-eval-dataset "Grab the ingredient box with gripper"
#   bash infer_pi05.sh my-eval-dataset "Grab the ingredient box with gripper" my-policy
#   bash infer_pi05.sh my-eval-dataset "Grab the ingredient box with gripper" my-policy /path/to/initial_pose.json

# INITIAL_POSE_FILE="initial_pose_lebai-gripper-black-taped-box-2_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_gripper_box"
# DEFAULT_SINGLE_TASK="Use gripper to pick up this black taped box"
# DEFAULT_POLICY_NAME="pi05_sjaj_gripper-box-plate-fdoor"

# INITIAL_POSE_FILE="initial_pose_lebai-gripper-plate_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_gripper_plate"
# DEFAULT_SINGLE_TASK="Use gripper to pick up this plate"
# DEFAULT_POLICY_NAME="pi05_sjaj_gripper-box-plate-fdoor"

# INITIAL_POSE_FILE="initial_pose_lebai-open-fridge-door_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_open-fridge"
# DEFAULT_SINGLE_TASK="open the door of the refrigerator"
# DEFAULT_POLICY_NAME="pi05_sjaj_gripper-box-plate-fdoor"

# DEFAULT_SINGLE_TASK="Use gripper to pick up this black taped box"
# DEFAULT_SINGLE_TASK="Use gripper to pick up this black taped box"
# DEFAULT_SINGLE_TASK="Grab the ingredient box with gripper"


INITIAL_POSE_FILE="initial_pose_lebai-gripper-black-taped-box-2_0.json"
DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_gripper_box2"
DEFAULT_SINGLE_TASK="Use gripper to pick up this black taped box"
# DEFAULT_SINGLE_TASK="Use the robotic gripper to lift the black taped box"
# DEFAULT_SINGLE_TASK="Engage grip and lift the black taped box"
DEFAULT_POLICY_NAME="pi05_sjaj_gripper-box-plate-fdoor"

# INITIAL_POSE_FILE="initial_pose_lebai-gripper-black-taped-box-2_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_gripper_box"
# DEFAULT_SINGLE_TASK="Use gripper to pick up this black taped box"
# # DEFAULT_SINGLE_TASK="Use the robotic gripper to lift the black taped box"
# # DEFAULT_SINGLE_TASK="Engage grip and lift the black taped box"
# DEFAULT_POLICY_NAME="pi05_sjaj_4tasks-hetero"

# INITIAL_POSE_FILE="initial_pose_lebai-gripper-plate_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_gripper_plate"
# DEFAULT_SINGLE_TASK="Use gripper to pick up this plate"
# DEFAULT_POLICY_NAME="pi05_sjaj_4tasks-hetero"

# INITIAL_POSE_FILE="initial_pose_lebai-open-fridge-door_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_open-fridge"
# DEFAULT_SINGLE_TASK="open the door of the refrigerator"
# DEFAULT_POLICY_NAME="pi05_sjaj_4tasks-hetero"

# INITIAL_POSE_FILE="initial_pose_lebai-suction-plate_0.json"
# DEFAULT_EVAL_DATASET_NAME="eval_lebai_pi05_suction-plate"
# DEFAULT_SINGLE_TASK="pick up the plate using suction cup"
# DEFAULT_POLICY_NAME="pi05_sjaj_4tasks-hetero"

DEFAULT_N_ACTION_STEPS=30
DEFAULT_NUM_EPISODES=10
DEFAULT_EPISODE_TIME_S=60
DEFAULT_RESET_TIME_S=60
DEFAULT_DEBUG=1
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-360p}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TORCHINDUCTOR_CACHE_DIR="${SCRIPT_DIR}/.torchinductor_cache"
export TRITON_CACHE_DIR="${SCRIPT_DIR}/.triton_cache"

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
DEFAULT_INITIAL_POSE_PATH="${SCRIPT_DIR}/$INITIAL_POSE_FILE"
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

if [[ -n "${HF_LEROBOT_HOME:-}" ]]; then
  LEROBOT_HOME="${HF_LEROBOT_HOME}"
elif [[ -n "${HF_HOME:-}" ]]; then
  LEROBOT_HOME="${HF_HOME}/lerobot"
else
  LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"
fi

DATASET_ROOT_PATH="${LEROBOT_HOME}/${DATASET_REPO_ID}"

EXTRA_ARGS=()
if [[ -n "${INITIAL_POSE_PATH}" ]]; then
  EXTRA_ARGS+=(--initial_pose_path="${INITIAL_POSE_PATH}")
fi
if [[ -d "${DATASET_ROOT_PATH}" ]]; then
  EXTRA_ARGS+=(--resume=true)
fi

bash setup_cams.sh "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}"

LEROBOT_DEBUG="${DEFAULT_DEBUG}" uv run --extra romoya --extra pi lerobot-record \
  --dataset.fps=30 \
  --dataset.vcodec=h264_nvenc \
  --trace_path=infer-pi05-trace.json \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
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
  --policy.gripper_threshold=0.5 \
  --teleop_end_effector_override=false \
  "${EXTRA_ARGS[@]}"
