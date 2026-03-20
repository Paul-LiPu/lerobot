#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash repair_model_schema.sh /path/to/model
#   bash repair_model_schema.sh /path/to/model my-dataset
#   bash repair_model_schema.sh /path/to/model my-dataset /path/to/output_model
#   bash repair_model_schema.sh /path/to/model "" /path/to/output_model
#
# Notes:
#   - Arg 1: model directory containing config.json and policy_*processor.json
#   - Arg 2: optional dataset name or full repo id used to read raw schemas
#   - Arg 3: optional output model dir; if omitted, patch in place
#   - HF_USER is detected automatically from `hf auth whoami` when Arg 2 is not a full repo id

if [[ $# -lt 1 ]]; then
  echo "Usage: bash repair_model_schema.sh /path/to/model [dataset-name-or-repo-id] [output-model-dir]" >&2
  exit 1
fi

MODEL_DIR="$1"
DATASET_NAME="${2:-}"
OUTPUT_DIR="${3:-}"

ARGS=(--model-dir "${MODEL_DIR}")

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${DATASET_NAME}" ]]; then
  HF_USER=$(
    hf auth whoami \
      | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
      | awk -F': *' 'NR==1 {print $2}'
  )

  if [[ -z "${HF_USER}" && "${DATASET_NAME}" != */* ]]; then
    echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
    exit 1
  fi

  if [[ "${DATASET_NAME}" == */* ]]; then
    DATASET_REPO_ID="${DATASET_NAME}"
  else
    DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"
  fi

  ARGS+=(--src-repo-id "${DATASET_REPO_ID}")
else
  ARGS+=(--use-default-lebai-schemas)
fi

uv run --extra romoya python repair_romoya_checkpoint_schema.py "${ARGS[@]}"
