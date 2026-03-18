#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train.sh
#   bash train.sh my-dataset
#   bash train.sh my-dataset my-policy
#   bash train.sh my-dataset my-policy my-run-name
#
# Notes:
#   - Arg 1: dataset name or full repo id
#   - Arg 2: policy repo name or full repo id
#   - Arg 3: local output/job name
#   - HF_USER is detected automatically from `hf auth whoami`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_act_config_abs.json}"

DEFAULT_DATASET_NAME="lebai-gripper-black-tape-box"
DEFAULT_POLICY_REPO_NAME="act_sjaj_lebai-gripper-black-tape-box"
DEFAULT_OUTPUT_NAME="act_sjaj_lebai-gripper-black-tape-box"
DEFAULT_POLICY_TYPE="act_romoya"
DEFAULT_DEVICE="cuda"
DEFAULT_STEPS=40000
DEFAULT_BATCH_SIZE=48
DEFAULT_NUM_WORKERS=12
DEFAULT_WANDB_ENABLE="true"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
POLICY_REPO_NAME="${2:-${DEFAULT_POLICY_REPO_NAME}}"
OUTPUT_NAME="${3:-${DEFAULT_OUTPUT_NAME}}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

if [[ "${DEFAULT_WANDB_ENABLE}" == "true" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    uv run wandb login --relogin "${WANDB_API_KEY}"
  elif [[ ! -f "${HOME}/.netrc" ]] || ! grep -q 'machine api\.wandb\.ai' "${HOME}/.netrc"; then
    echo "wandb is not logged in. Starting wandb login..."
    uv run wandb login
  fi
fi

if [[ "${DATASET_NAME}" == */* ]]; then
  DATASET_REPO_ID="${DATASET_NAME}"
else
  DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"
fi

if [[ "${POLICY_REPO_NAME}" == */* ]]; then
  POLICY_REPO_ID="${POLICY_REPO_NAME}"
else
  POLICY_REPO_ID="${HF_USER}/${POLICY_REPO_NAME}"
fi

OUTPUT_DIR="outputs/train/${OUTPUT_NAME}"
JOB_NAME="${OUTPUT_NAME}"

ROMOYA_PREPARED_DATASET_REPO_ID="${ROMOYA_PREPARED_DATASET_REPO_ID:-${DATASET_REPO_ID}_romoya_prepared}"
ROMOYA_PREPARED_DATASET_ROOT="${ROMOYA_PREPARED_DATASET_ROOT:-}"
DATASET_ROOT_CLI_ARGS=()
if [[ -n "${ROMOYA_PREPARED_DATASET_ROOT}" ]]; then
  DATASET_ROOT_CLI_ARGS+=(--dataset.root="${ROMOYA_PREPARED_DATASET_ROOT}")
fi

ROMOYA_PREPARE_MODE="$(
  python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
policy = payload.get("policy", payload)
if policy.get("type") != "act_romoya":
    print("none")
    raise SystemExit(0)

action_mode = policy.get("action_mode", "__missing__")
if action_mode is not None and action_mode != "__missing__":
    print("legacy")
    raise SystemExit(0)

binary_state = policy.get("binary_state") or []
binary_action = policy.get("binary_action") or []
delta_action = policy.get("delta_action") or []
needs_prepare = any(value is not None for value in binary_state) or any(
    value is not None for value in binary_action
) or any(bool(value) for value in delta_action)
print("generic" if needs_prepare else "none")
PY
)"

if [[ "${ROMOYA_PREPARE_MODE}" != "none" ]]; then
  if [[ "${DATASET_REPO_ID}" == *_romoya_prepared ]]; then
    echo "Using already-prepared Romoya dataset: ${DATASET_REPO_ID}"
  else
    PREPARED_MATCH="$(
      python3 - "$CONFIG_PATH" "$DATASET_REPO_ID" "$ROMOYA_PREPARED_DATASET_REPO_ID" "$ROMOYA_PREPARED_DATASET_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
src_repo_id = sys.argv[2]
dst_repo_id = sys.argv[3]
dst_root = sys.argv[4] or None

payload = json.loads(config_path.read_text())
policy = payload.get("policy", payload)
expected = {
    "action_mode": policy.get("action_mode"),
    "state_feature_names": policy.get("state_feature_names"),
    "action_feature_names": policy.get("action_feature_names"),
    "binary_state": policy.get("binary_state"),
    "binary_action": policy.get("binary_action"),
    "delta_action": policy.get("delta_action"),
}

if dst_root:
    base_root = Path(dst_root)
else:
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
    base_root = hf_lerobot_home / dst_repo_id
info_path = base_root / "meta" / "info.json"
stats_path = base_root / "meta" / "stats.json"
if not info_path.is_file() or not stats_path.is_file():
    print("missing")
    raise SystemExit(0)

info = json.loads(info_path.read_text())
romoya_prepare = info.get("romoya_prepare") or {}
if romoya_prepare.get("source_repo_id") != src_repo_id:
    print("stale")
    raise SystemExit(0)

def eq(key):
    return romoya_prepare.get(key) == expected.get(key)

if expected["action_mode"] is not None:
    is_match = eq("action_mode")
else:
    is_match = all(
        eq(key)
        for key in (
            "action_mode",
            "state_feature_names",
            "action_feature_names",
            "binary_state",
            "binary_action",
            "delta_action",
        )
    )

if is_match:
    print("match")
else:
    print("stale")
PY
    )"

    if [[ "${PREPARED_MATCH}" != "match" ]]; then
      echo "Preparing Romoya training dataset: ${ROMOYA_PREPARED_DATASET_REPO_ID}"
      PREPARE_ARGS=(
        --src-repo-id "${DATASET_REPO_ID}"
        --dst-repo-id "${ROMOYA_PREPARED_DATASET_REPO_ID}"
        --config-path "${CONFIG_PATH}"
      )
      if [[ -n "${ROMOYA_PREPARED_DATASET_ROOT}" ]]; then
        PREPARE_ARGS+=(--dst-root "${ROMOYA_PREPARED_DATASET_ROOT}")
      fi
      uv run --extra romoya python prepare_romoya_dataset.py "${PREPARE_ARGS[@]}"
    else
      echo "Reusing prepared Romoya dataset: ${ROMOYA_PREPARED_DATASET_REPO_ID}"
    fi

    DATASET_REPO_ID="${ROMOYA_PREPARED_DATASET_REPO_ID}"
  fi
fi

python3 - "$DATASET_REPO_ID" "${ROMOYA_PREPARED_DATASET_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

dataset_repo_id = sys.argv[1]
dataset_root_override = sys.argv[2] or None
if dataset_root_override:
    dataset_root = Path(dataset_root_override)
else:
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
    dataset_root = hf_lerobot_home / dataset_repo_id
info_path = dataset_root / "meta" / "info.json"
stats_path = dataset_root / "meta" / "stats.json"

if not info_path.is_file() or not stats_path.is_file():
    print(f"Dataset stats not found locally under {dataset_root}; skipping stats print.")
    raise SystemExit(0)

info = json.loads(info_path.read_text())
stats = json.loads(stats_path.read_text())
features = info.get("features", {})

def print_vector_stats(key: str) -> None:
    feature = features.get(key)
    stat = stats.get(key)
    if feature is None or stat is None:
        print(f"{key}: not found")
        return
    names = feature.get("names") or [f"{key}[{i}]" for i in range(len(stat.get("mean", [])))]
    print(f"{key} stats:")
    for i, name in enumerate(names):
        mean = stat["mean"][i]
        std = stat["std"][i]
        min_val = stat["min"][i]
        max_val = stat["max"][i]
        print(f"  {i}: {name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")

print(f"Training dataset: {dataset_repo_id}")
print(f"Dataset root: {dataset_root}")
romoya_prepare = info.get("romoya_prepare")
if romoya_prepare is not None:
    print(f"Prepared dataset source: {romoya_prepare.get('source_repo_id')}")
print_vector_stats("observation.state")
print_vector_stats("action")
PY

uv run --extra romoya lerobot-train \
  --config_path="${CONFIG_PATH}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  "${DATASET_ROOT_CLI_ARGS[@]}" \
  --policy.type="${DEFAULT_POLICY_TYPE}" \
  --policy.device="${DEFAULT_DEVICE}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${DEFAULT_STEPS}" \
  --batch_size="${DEFAULT_BATCH_SIZE}" \
  --num_workers="${DEFAULT_NUM_WORKERS}" \
  --wandb.enable="${DEFAULT_WANDB_ENABLE}"
