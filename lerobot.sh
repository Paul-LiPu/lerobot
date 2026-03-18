#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash lerobot.sh list
#   bash lerobot.sh rm lebai-suction-plate
#   bash lerobot.sh info task lebai-suction-plate
#   bash lerobot.sh info dir lebai-suction-plate
#   bash lerobot.sh info camera lebai-suction-plate
#   bash lerobot.sh info task PL2011/lebai-suction-plate
#   bash lerobot.sh info task /home/cvlab/.cache/huggingface/lerobot/PL2011/lebai-suction-plate
#
# Notes:
#   - Dataset input can be a dataset name, full repo id, or local dataset path
#   - Plain dataset names are resolved under HF_USER from `hf auth whoami`

DEFAULT_HF_HOME="${HOME}/.cache/huggingface"
CACHE_ROOT="${HF_LEROBOT_HOME:-${HF_HOME:-${DEFAULT_HF_HOME}}/lerobot}"

print_usage() {
  cat <<'EOF'
Usage:
  bash lerobot.sh list
  bash lerobot.sh rm <dataset>
  bash lerobot.sh info task <dataset>
  bash lerobot.sh info dir <dataset>
  bash lerobot.sh info camera <dataset>

Examples:
  bash lerobot.sh list
  bash lerobot.sh rm lebai-suction-plate
  bash lerobot.sh info task lebai-suction-plate
  bash lerobot.sh info dir PL2011/lebai-suction-plate
  bash lerobot.sh info camera /home/cvlab/.cache/huggingface/lerobot/PL2011/lebai-suction-plate
EOF
}

get_hf_user() {
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
}

resolve_dataset_root() {
  local dataset_input="$1"

  if [[ -z "${dataset_input}" ]]; then
    echo "Dataset is required." >&2
    exit 1
  fi

  if [[ "${dataset_input}" == /* ]]; then
    printf '%s\n' "${dataset_input}"
    return 0
  fi

  if [[ "${dataset_input}" == */* ]]; then
    printf '%s/%s\n' "${CACHE_ROOT}" "${dataset_input}"
    return 0
  fi

  local hf_user
  hf_user="$(get_hf_user)"
  if [[ -z "${hf_user}" ]]; then
    echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
    exit 1
  fi

  printf '%s/%s/%s\n' "${CACHE_ROOT}" "${hf_user}" "${dataset_input}"
}

require_dataset_root() {
  local dataset_root="$1"
  if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset directory not found: ${dataset_root}" >&2
    exit 1
  fi
  if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
    echo "Dataset metadata not found: ${dataset_root}/meta/info.json" >&2
    exit 1
  fi
}

list_datasets() {
  if [[ ! -d "${CACHE_ROOT}" ]]; then
    echo "Local LeRobot cache does not exist: ${CACHE_ROOT}" >&2
    exit 1
  fi

  python3 - "${CACHE_ROOT}" <<'PY'
import sys
from pathlib import Path

cache_root = Path(sys.argv[1]).expanduser()
datasets = []

for info_path in sorted(cache_root.glob("*/*/meta/info.json")):
    dataset_root = info_path.parent.parent
    repo_id = str(dataset_root.relative_to(cache_root))
    datasets.append((repo_id, str(dataset_root)))

if not datasets:
    print("No local datasets found.")
    raise SystemExit(0)

for repo_id, dataset_root in datasets:
    print(f"{repo_id}\t{dataset_root}")
PY
}

show_task_info() {
  local dataset_root="$1"
  local tasks_path="${dataset_root}/meta/tasks.parquet"

  if [[ ! -f "${tasks_path}" ]]; then
    echo "tasks.parquet not found: ${tasks_path}" >&2
    exit 1
  fi

  python3 - "${tasks_path}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

tasks_path = Path(sys.argv[1])
df = pd.read_parquet(tasks_path)

if "task" in df.columns:
    tasks = df["task"].tolist()
else:
    tasks = df.index.tolist()

for task in tasks:
    print(task)
PY
}

show_camera_info() {
  local dataset_root="$1"
  local info_path="${dataset_root}/meta/info.json"
  local camera_settings_path="${dataset_root}/meta/camera-settings.json"

  python3 - "${info_path}" "${camera_settings_path}" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
camera_settings_path = Path(sys.argv[2])

with info_path.open() as f:
    info = json.load(f)

features = info.get("features", {})
fps = info.get("fps")
video_features = [
    (key, spec) for key, spec in features.items()
    if spec.get("dtype") == "video" or key.startswith("observation.images.")
]

if not video_features:
    print("No camera/video features found in meta/info.json")
else:
    print("Dataset camera features:")
    for key, spec in video_features:
        shape = spec.get("shape")
        feature_fps = spec.get("fps", fps)
        line = f"- {key}"
        if shape is not None:
            line += f": shape={shape}"
        if feature_fps is not None:
            line += f", fps={feature_fps}"
        print(line)

if not camera_settings_path.exists():
    raise SystemExit(0)

print("")
print("Saved V4L2 camera settings:")
with camera_settings_path.open() as f:
    settings = json.load(f)

if "devices" in settings:
    devices = settings["devices"]
else:
    devices = {settings.get("device", "unknown"): settings}

for device_path, device in devices.items():
    display_name = device.get("display_name")
    stream = device.get("stream", {})
    line = f"- {device_path}"
    if display_name:
        line += f" ({display_name})"
    print(line)
    if stream:
        width = stream.get("width")
        height = stream.get("height")
        pixel_format = stream.get("pixel_format")
        stream_fps = stream.get("fps")
        details = []
        if width is not None and height is not None:
            details.append(f"{width}x{height}")
        if pixel_format:
            details.append(str(pixel_format))
        if stream_fps is not None:
            details.append(f"{stream_fps} fps")
        if details:
            print(f"  stream: {', '.join(details)}")
PY
}

remove_dataset() {
  local dataset_root="$1"
  if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset directory not found: ${dataset_root}" >&2
    exit 1
  fi
  rm -rf "${dataset_root}"
  printf 'Removed %s\n' "${dataset_root}"
}

main() {
  local command="${1:-}"

  case "${command}" in
    list)
      if [[ $# -ne 1 ]]; then
        print_usage
        exit 1
      fi
      list_datasets
      ;;
    rm)
      local dataset_input="${2:-}"
      local dataset_root

      if [[ $# -ne 2 ]]; then
        print_usage
        exit 1
      fi

      dataset_root="$(resolve_dataset_root "${dataset_input}")"
      remove_dataset "${dataset_root}"
      ;;
    info)
      local field="${2:-}"
      local dataset_input="${3:-}"
      local dataset_root

      case "${field}" in
        task|dir|camera)
          ;;
        *)
          print_usage
          exit 1
          ;;
      esac

      dataset_root="$(resolve_dataset_root "${dataset_input}")"
      require_dataset_root "${dataset_root}"

      case "${field}" in
        task)
          show_task_info "${dataset_root}"
          ;;
        dir)
          printf '%s\n' "${dataset_root}"
          ;;
        camera)
          show_camera_info "${dataset_root}"
          ;;
      esac
      ;;
    -h|--help|help|"")
      print_usage
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
