#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash teleop.sh
#
# Notes:
#   - Starts Lebai leader-follower teleoperation with the current camera setup
#   - Camera paths are expected at /dev/cam_wrist, /dev/cam_top, and /dev/cam_side

uv run --extra romoya lerobot-teleoperate \
  --fps=30 \
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
  --display_data=true
