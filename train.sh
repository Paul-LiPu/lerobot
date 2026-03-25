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
HELPER_DIR="${SCRIPT_DIR}/lerobot_robot_romoya/helper"

CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_act_config_abs_cg.json}"
DEFAULT_DATASET_REPOS="PL2011/lebai-gripper-black-taped-box-2"
DEFAULT_POLICY_REPO_NAME="act_sjaj_gripper-box"
DEFAULT_OUTPUT_NAME="act_sjaj_gripper"
DEFAULT_POLICY_TYPE="act_romoya"
DEFAULT_DEVICE="cuda"
DEFAULT_STEPS=20000
DEFAULT_BATCH_SIZE=48

# CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SCRIPT_DIR}/train_pi05_romoya_config.json}"
# DEFAULT_DATASET_REPOS=(
#   "PL2011/lebai-open-fridge-door"
#   "PL2011/lebai-gripper-plate"
#   "PL2011/lebai-gripper-black-taped-box-2"
# )
# DEFAULT_POLICY_REPO_NAME="pi05_sjaj_gripper-box-plate-fdoor"
# DEFAULT_OUTPUT_NAME="pi05_sjaj_gripper-box-plate-fdoor"
# DEFAULT_POLICY_TYPE="pi05_romoya"
# DEFAULT_DEVICE="cuda"
# DEFAULT_STEPS=40000
# DEFAULT_BATCH_SIZE=24


DEFAULT_NUM_WORKERS=8
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
    python3 "${HELPER_DIR}/check_merged_dataset.py" \
      "${DATASET_REPO_ID}" \
      "${ROMOYA_MERGED_DATASET_ROOT}" \
      "${DATASET_REPO_IDS[@]}"
  )"

  if [[ "${MERGED_MATCH}" != "match" ]]; then
    echo "Merging training datasets into ${DATASET_REPO_ID}"
    uv run "${UV_EXTRA_ARGS[@]}" python "${HELPER_DIR}/merge_datasets.py" \
      "${DATASET_REPO_ID}" \
      "${ROMOYA_MERGED_DATASET_ROOT}" \
      "${DATASET_REPO_IDS[@]}"
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

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "Output directory ${OUTPUT_DIR} already exists and resume is false. Choose a new output name or remove the existing directory before training." >&2
  exit 1
fi

destroy_vast_instance_if_configured() {
  if [[ -z "${VAST_CONTAINERLABEL:-}" ]]; then
    return
  fi

  if ! command -v vastai >/dev/null 2>&1; then
    echo "VAST_CONTAINERLABEL is set, but 'vastai' CLI is not available; skipping instance destroy." >&2
    return
  fi

  local instance_id
  instance_id="$(echo "${VAST_CONTAINERLABEL}" | sed 's/^C\.//')"
  if [[ -z "${instance_id}" ]]; then
    echo "Failed to determine Vast instance id from VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL}'; skipping destroy." >&2
    return
  fi

  echo "Destroying Vast instance ${instance_id}"
  vastai destroy instance "${instance_id}"
}

ROMOYA_PREPARED_DATASET_REPO_ID="${ROMOYA_PREPARED_DATASET_REPO_ID:-${DATASET_REPO_ID}_romoya_prepared}"
ROMOYA_PREPARED_DATASET_ROOT="${ROMOYA_PREPARED_DATASET_ROOT:-}"

ROMOYA_PREPARE_MODE="$(
  python3 "${HELPER_DIR}/detect_prepare_mode.py" "$CONFIG_PATH"
)"

if [[ "${ROMOYA_PREPARE_MODE}" != "none" ]]; then
  if [[ "${DATASET_REPO_ID}" == *_romoya_prepared ]]; then
    echo "Using already-prepared Romoya dataset: ${DATASET_REPO_ID}"
  else
    PREPARED_MATCH="$(
      python3 "${HELPER_DIR}/check_prepared_dataset.py" \
        "$CONFIG_PATH" \
        "$DATASET_REPO_ID" \
        "$ROMOYA_PREPARED_DATASET_REPO_ID" \
        "$ROMOYA_PREPARED_DATASET_ROOT"
    )"

    if [[ "${PREPARED_MATCH}" != "match" ]]; then
      echo "Preparing Romoya training dataset: ${ROMOYA_PREPARED_DATASET_REPO_ID}"
      PREPARE_ARGS=(
        --src-repo-id "${DATASET_REPO_ID}"
        --dst-repo-id "${ROMOYA_PREPARED_DATASET_REPO_ID}"
        --config-path "${CONFIG_PATH}"
      )
      RESIZE_ARGS=()
      mapfile -t RESIZE_ARGS < <(python3 "${HELPER_DIR}/get_prepare_resize_args.py" "$CONFIG_PATH")
      if [[ "${#RESIZE_ARGS[@]}" -gt 0 ]]; then
        PREPARE_ARGS+=("${RESIZE_ARGS[@]}")
      fi
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

python3 "${HELPER_DIR}/print_dataset_stats.py" \
  "$DATASET_REPO_ID" \
  "${FINAL_DATASET_ROOT}" \
  "${DATASET_REPO_IDS[@]}"

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

destroy_vast_instance_if_configured
