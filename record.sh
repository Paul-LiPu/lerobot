#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash record.sh
#   bash record.sh my-dataset
#   bash record.sh my-dataset "Grab the black cube"
#   bash record.sh my-dataset "Grab the black cube" /path/to/initial_pose.json
#   bash record.sh my-dataset "Grab the black cube" /path/to/initial_pose.json 82
#
# Notes:
#   - Arg 1: dataset name
#   - Arg 2: task string
#   - Arg 3: initial pose json path
#   - Arg 4: closed gripper position
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_DATASET_NAME="record-test"
DEFAULT_SINGLE_TASK="Grab the black cube"
DEFAULT_GRIPPER_CLOSED_POSITION="0"
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-1080p}"

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

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
SINGLE_TASK="${2:-${DEFAULT_SINGLE_TASK}}"
INITIAL_POSE_PATH_ARG="${3:-}"
GRIPPER_CLOSED_POSITION="${4:-${GRIPPER_CLOSED_POSITION:-${DEFAULT_GRIPPER_CLOSED_POSITION}}}"

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
LOCAL_DATASET_DIR="${HOME}/.cache/huggingface/lerobot/${DATASET_REPO_ID}"
LOCAL_DATASET_META_DIR="${LOCAL_DATASET_DIR}/meta"
DEFAULT_INITIAL_POSE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/initial_pose.json"
INITIAL_POSE_PATH="${INITIAL_POSE_PATH_ARG:-${INITIAL_POSE_PATH:-${DEFAULT_INITIAL_POSE_PATH}}}"
TMP_CAMERA_SETTINGS="$(mktemp "${TMPDIR:-/tmp}/camera-settings.XXXXXX.json")"

cleanup() {
  rm -f "${TMP_CAMERA_SETTINGS}"
}
trap cleanup EXIT

EXTRA_ARGS=()
if [[ -n "${INITIAL_POSE_PATH}" ]]; then
  EXTRA_ARGS+=(--initial_pose_path="${INITIAL_POSE_PATH}")
fi
if [[ -d "${LOCAL_DATASET_DIR}" ]]; then
  EXTRA_ARGS+=(--resume=true)
fi

bash setup_cams.sh "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}"

uv run lerobot-v4l2-camera-settings save \
  --device /dev/cam_wrist \
  --device /dev/cam_top \
  --device /dev/cam_side \
  --output "${TMP_CAMERA_SETTINGS}"
  # --resume=true \
uv run --extra romoya lerobot-record \
  --trace_path=record-loop-trace.json \
  --dataset.fps=30 \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: /dev/cam_top, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: /dev/cam_side, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}}" \
  --robot.gripper_closed_position="${GRIPPER_CLOSED_POSITION}" \
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
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=300 \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.streaming_encoding=true \
  --dataset.vcodec=h264_nvenc \
  --dataset.encoder_threads=1 \
  "${EXTRA_ARGS[@]}"

mkdir -p "${LOCAL_DATASET_META_DIR}"
cp "${TMP_CAMERA_SETTINGS}" "${LOCAL_DATASET_META_DIR}/camera-settings.json"
hf upload \
  "${DATASET_REPO_ID}" \
  "${LOCAL_DATASET_META_DIR}/camera-settings.json" \
  "meta/camera-settings.json" \
  --repo-type dataset
