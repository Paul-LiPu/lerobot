#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train.sh
#   bash train.sh my-dataset
#   bash train.sh repo1,repo2,repo3
#   bash train.sh my-dataset my-policy
#   bash train.sh my-dataset my-policy my-run-name
#
# Notes:
#   - Arg 1: dataset name, full repo id, or comma-separated repo ids
#   - Arg 2: policy repo name or full repo id
#   - Arg 3: local output/job name
#   - HF_USER is detected automatically from `hf auth whoami`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_act_config_abs.json}"
CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_pi05_romoya_config.json}"

DEFAULT_DATASET_REPOS=(
  "PL2011/lebai-open-fridge-door"
  "PL2011/lebai-gripper-plate"
  "PL2011/lebai-gripper-black-taped-box-2"
)
DEFAULT_POLICY_REPO_NAME="pi05_sjaj_gripper-box-plate-fdoor"
DEFAULT_OUTPUT_NAME="pi05_sjaj_gripper-box-plate-fdoor"
DEFAULT_POLICY_TYPE="pi05_romoya"
DEFAULT_DEVICE="cuda"
DEFAULT_STEPS=40000
DEFAULT_BATCH_SIZE=24
DEFAULT_NUM_WORKERS=12
DEFAULT_WANDB_ENABLE="true"

DATASET_SPEC="${1:-}"
POLICY_REPO_NAME="${2:-${DEFAULT_POLICY_REPO_NAME}}"
OUTPUT_NAME="${3:-${DEFAULT_OUTPUT_NAME}}"

UV_EXTRA_ARGS=(--extra romoya)
if [[ "${DEFAULT_POLICY_TYPE}" == "pi05" || "${DEFAULT_POLICY_TYPE}" == "pi05_romoya" ]]; then
  UV_EXTRA_ARGS+=(--extra pi)
fi

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

resolve_repo_id() {
  local value="$1"
  if [[ "${value}" == */* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${HF_USER}/${value}"
  fi
}

DATASET_INPUTS=()
if [[ -z "${DATASET_SPEC}" ]]; then
  DATASET_INPUTS=("${DEFAULT_DATASET_REPOS[@]}")
elif [[ "${DATASET_SPEC}" == *,* ]]; then
  IFS=',' read -r -a DATASET_INPUTS <<< "${DATASET_SPEC}"
else
  DATASET_INPUTS=("${DATASET_SPEC}")
fi

DATASET_REPO_IDS=()
for dataset_input in "${DATASET_INPUTS[@]}"; do
  dataset_input="${dataset_input//[[:space:]]/}"
  if [[ -z "${dataset_input}" ]]; then
    continue
  fi
  DATASET_REPO_IDS+=("$(resolve_repo_id "${dataset_input}")")
done

if [[ "${#DATASET_REPO_IDS[@]}" -eq 0 ]]; then
  echo "No datasets resolved for training." >&2
  exit 1
fi

ROMOYA_MERGED_DATASET_ROOT="${ROMOYA_MERGED_DATASET_ROOT:-}"
FINAL_DATASET_ROOT=""

if [[ "${#DATASET_REPO_IDS[@]}" -eq 1 ]]; then
  DATASET_REPO_ID="${DATASET_REPO_IDS[0]}"
else
  DATASET_REPO_ID="${ROMOYA_MERGED_DATASET_REPO_ID:-${HF_USER}/dataset_merged}"

  MERGED_MATCH="$(
    python3 - "${DATASET_REPO_ID}" "${ROMOYA_MERGED_DATASET_ROOT}" "${DATASET_REPO_IDS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

merged_repo_id = sys.argv[1]
merged_root_override = sys.argv[2] or None
source_repo_ids = sys.argv[3:]

if merged_root_override:
    dataset_root = Path(merged_root_override)
else:
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
    dataset_root = hf_lerobot_home / merged_repo_id

info_path = dataset_root / "meta" / "info.json"
stats_path = dataset_root / "meta" / "stats.json"
if not info_path.is_file() or not stats_path.is_file():
    print("missing")
    raise SystemExit(0)

info = json.loads(info_path.read_text())
romoya_merge = info.get("romoya_merge") or {}
if romoya_merge.get("source_repo_ids") == source_repo_ids:
    print("match")
else:
    print("stale")
PY
  )"

  if [[ "${MERGED_MATCH}" != "match" ]]; then
    echo "Merging training datasets into ${DATASET_REPO_ID}"
    uv run "${UV_EXTRA_ARGS[@]}" python - "${DATASET_REPO_ID}" "${ROMOYA_MERGED_DATASET_ROOT}" "${DATASET_REPO_IDS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset

merged_repo_id = sys.argv[1]
merged_root_override = sys.argv[2] or None
source_repo_ids = sys.argv[3:]

datasets = [LeRobotDataset(repo_id) for repo_id in source_repo_ids]
merged_dataset = merge_datasets(
    datasets,
    output_repo_id=merged_repo_id,
    output_dir=Path(merged_root_override) if merged_root_override else None,
)
merged_dataset.meta.info["romoya_merge"] = {
    "source_repo_ids": source_repo_ids,
}
write_info(merged_dataset.meta.info, merged_dataset.root)
PY
  else
    echo "Reusing merged dataset: ${DATASET_REPO_ID}"
  fi

  if [[ -n "${ROMOYA_MERGED_DATASET_ROOT}" ]]; then
    FINAL_DATASET_ROOT="${ROMOYA_MERGED_DATASET_ROOT}"
  fi
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

ROMOYA_PREPARE_MODE="$(
  python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
policy = payload.get("policy", payload)
if policy.get("type") not in {"act_romoya", "pi05_romoya"}:
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
if not romoya_prepare.get("source_raw_observation_state_feature_names") or not romoya_prepare.get("source_raw_action_feature_names"):
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
      if [[ -n "${FINAL_DATASET_ROOT}" ]]; then
        PREPARE_ARGS+=(--src-root "${FINAL_DATASET_ROOT}")
      fi
      if [[ -n "${ROMOYA_PREPARED_DATASET_ROOT}" ]]; then
        PREPARE_ARGS+=(--dst-root "${ROMOYA_PREPARED_DATASET_ROOT}")
      fi
      uv run "${UV_EXTRA_ARGS[@]}" python prepare_romoya_dataset.py "${PREPARE_ARGS[@]}"
    else
      echo "Reusing prepared Romoya dataset: ${ROMOYA_PREPARED_DATASET_REPO_ID}"
    fi

    DATASET_REPO_ID="${ROMOYA_PREPARED_DATASET_REPO_ID}"
    if [[ -n "${ROMOYA_PREPARED_DATASET_ROOT}" ]]; then
      FINAL_DATASET_ROOT="${ROMOYA_PREPARED_DATASET_ROOT}"
    else
      FINAL_DATASET_ROOT=""
    fi
  fi
fi

DATASET_ROOT_CLI_ARGS=()
if [[ -n "${FINAL_DATASET_ROOT}" ]]; then
  DATASET_ROOT_CLI_ARGS+=(--dataset.root="${FINAL_DATASET_ROOT}")
fi

python3 - "$DATASET_REPO_ID" "${FINAL_DATASET_ROOT}" "${DATASET_REPO_IDS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

dataset_repo_id = sys.argv[1]
dataset_root_override = sys.argv[2] or None
source_repo_ids = sys.argv[3:]
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
if source_repo_ids:
    print("Source datasets:")
    for repo_id in source_repo_ids:
        print(f"  - {repo_id}")
romoya_prepare = info.get("romoya_prepare")
if romoya_prepare is not None:
    print(f"Prepared dataset source: {romoya_prepare.get('source_repo_id')}")
print_vector_stats("observation.state")
print_vector_stats("action")
PY

uv run "${UV_EXTRA_ARGS[@]}" lerobot-train \
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
