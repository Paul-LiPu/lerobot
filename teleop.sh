#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash teleop.sh
#
# Notes:
#   - Starts Lebai leader-follower teleoperation with the current camera setup
#   - Camera paths are expected at /dev/cam_wrist, /dev/cam_top, and /dev/cam_side

CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-360p}"
GRIPPER_OPEN_POSITION="${GRIPPER_OPEN_POSITION:-99}"

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

bash setup_cams.sh "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}"

uv run --extra romoya lerobot-teleoperate \
  --fps=30 \
  --robot.type=romoya_lebai_follower \
  --robot.ip=192.168.50.172 \
  --robot.id=lebai_follower_arm \
  --robot.trace_path=teleop-trace.json \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: /dev/cam_top, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: /dev/cam_side, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: 30, fourcc: MJPG}}" \
  --robot.gripper_open_position=60 \
  --robot.gripper_closed_position=0 \
  --robot.gripper_force=100 \
  --robot.acceleration=1.0 \
  --robot.velocity=1.0 \
  --robot.blend_radius=0.0 \
  --teleop.type=romoya_lebai_leader \
  --teleop.ip=192.168.50.154 \
  --teleop.gripper_force=100 \
  --teleop.id=lebai_leader_arm \
  --display_data=true
