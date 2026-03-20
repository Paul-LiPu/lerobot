#!/usr/bin/env bash
set -euo pipefail

CAMERAS=(
  /dev/cam_wrist
  /dev/cam_top
  /dev/cam_side
)

WIDTH="${1:-1920}"
HEIGHT="${2:-1080}"
FPS=30
PIXEL_FORMAT=MJPG
FOCUS_VALUE=0

configure_camera() {
  local cam="$1"

  echo "Configuring ${cam} to ${WIDTH}x${HEIGHT} ${PIXEL_FORMAT} ${FPS}fps ..."

  v4l2-ctl --device="${cam}" \
    --set-fmt-video=width=${WIDTH},height=${HEIGHT},pixelformat=${PIXEL_FORMAT}

  v4l2-ctl --device="${cam}" --set-parm=${FPS}

  local ctrls
  ctrls="$(v4l2-ctl --device="${cam}" -L 2>/dev/null || true)"

  if echo "${ctrls}" | grep -q '^[[:space:]]*focus_automatic_continuous'; then
    v4l2-ctl --device="${cam}" --set-ctrl=focus_automatic_continuous=0
    echo "  autofocus: focus_automatic_continuous=0"
  elif echo "${ctrls}" | grep -q '^[[:space:]]*focus_auto'; then
    v4l2-ctl --device="${cam}" --set-ctrl=focus_auto=0
    echo "  autofocus: focus_auto=0"
  else
    echo "  autofocus: no supported autofocus control found"
  fi

  if echo "${ctrls}" | grep -q '^[[:space:]]*focus_absolute'; then
    v4l2-ctl --device="${cam}" --set-ctrl=focus_absolute=${FOCUS_VALUE}
    echo "  focus_absolute=${FOCUS_VALUE}"
  else
    echo "  focus_absolute: unsupported"
  fi
}

verify_camera() {
  local cam="$1"
  local fmt
  local parm
  local ctrls

  fmt="$(v4l2-ctl --device="${cam}" --get-fmt-video)"
  parm="$(v4l2-ctl --device="${cam}" --get-parm || true)"
  ctrls="$(v4l2-ctl --device="${cam}" -L 2>/dev/null || true)"

  local pass_width="FAIL"
  local pass_height="FAIL"
  local pass_format="FAIL"
  local pass_fps="FAIL"
  local pass_af="FAIL"
  local pass_focus="FAIL"

  echo "${fmt}" | grep -q "Width/Height[[:space:]]*: ${WIDTH}/${HEIGHT}" && pass_width="PASS" && pass_height="PASS"
  echo "${fmt}" | grep -q "Pixel Format[[:space:]]*: '${PIXEL_FORMAT}'" && pass_format="PASS"
  echo "${parm}" | grep -q "Frames per second:[[:space:]]*${FPS}\.000" && pass_fps="PASS"

  if echo "${ctrls}" | grep -q '^[[:space:]]*focus_automatic_continuous'; then
    echo "${ctrls}" | grep -q '^[[:space:]]*focus_automatic_continuous .*value=0' && pass_af="PASS"
  elif echo "${ctrls}" | grep -q '^[[:space:]]*focus_auto'; then
    echo "${ctrls}" | grep -q '^[[:space:]]*focus_auto .*value=0' && pass_af="PASS"
  else
    pass_af="N/A"
  fi

  if echo "${ctrls}" | grep -q '^[[:space:]]*focus_absolute'; then
    echo "${ctrls}" | grep -q "^[[:space:]]*focus_absolute .*value=${FOCUS_VALUE}\\b" && pass_focus="PASS"
  else
    pass_focus="N/A"
  fi

  echo "============================================================"
  echo "DEVICE: ${cam}"
  echo "------------------------------------------------------------"
  echo "[Format]"
  echo "${fmt}"
  echo
  echo "[Streaming Parameters]"
  echo "${parm}"
  echo
  echo "[Focus Controls]"
  echo "${ctrls}" | grep -E '^[[:space:]]*focus_automatic_continuous|^[[:space:]]*focus_auto|^[[:space:]]*focus_absolute' || true
  echo
  echo "[Verification]"
  echo "  width/height: ${pass_width}"
  echo "  pixel format: ${pass_format}"
  echo "  fps:          ${pass_fps}"
  echo "  autofocus:    ${pass_af}"
  echo "  focus value:  ${pass_focus}"
  echo
}

for cam in "${CAMERAS[@]}"; do
  configure_camera "${cam}"
done

echo
echo "Verification summary (${WIDTH}x${HEIGHT})"
echo

for cam in "${CAMERAS[@]}"; do
  verify_camera "${cam}"
done
