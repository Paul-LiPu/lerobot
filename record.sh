#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATASET_NAME="record-test"
DEFAULT_SINGLE_TASK="Grab the black cube"
DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
SINGLE_TASK="${2:-${DEFAULT_SINGLE_TASK}}"

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
TMP_CAMERA_SETTINGS="$(mktemp "${TMPDIR:-/tmp}/camera-settings.XXXXXX.json")"

cleanup() {
  rm -f "${TMP_CAMERA_SETTINGS}"
}
trap cleanup EXIT

uv run lerobot-v4l2-camera-settings save \
  --device /dev/cam_wrist \
  --device /dev/cam_top \
  --device /dev/cam_side \
  --output "${TMP_CAMERA_SETTINGS}"
  # --resume=true \
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
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=120 \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=4

mkdir -p "${LOCAL_DATASET_META_DIR}"
cp "${TMP_CAMERA_SETTINGS}" "${LOCAL_DATASET_META_DIR}/camera-settings.json"
hf upload \
  "${DATASET_REPO_ID}" \
  "${LOCAL_DATASET_META_DIR}/camera-settings.json" \
  "meta/camera-settings.json" \
  --repo-type dataset
