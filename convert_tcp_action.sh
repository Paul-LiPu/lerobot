#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash convert_tcp_action.sh
#   bash convert_tcp_action.sh lebai-suction-plate
#   bash convert_tcp_action.sh lebai-suction-plate lebai-suction-plate-tcp-action
#   bash convert_tcp_action.sh lebai-suction-plate lebai-suction-plate-tcp-action true
#   bash convert_tcp_action.sh PL2011/lebai-suction-plate PL2011/lebai-suction-plate-tcp-action
#
# Notes:
#   - Arg 1: source dataset name or full repo id
#   - Arg 2: destination dataset name or full repo id
#   - Arg 3: whether to overwrite existing tcp.* fields in the source action (`true` or `false`)
#   - HF_USER is detected automatically from `hf auth whoami`

DEFAULT_DATASET_NAME="lebai-suction-plate"
DEFAULT_OUTPUT_SUFFIX="-tcp-action"

DATASET_NAME="${1:-${DEFAULT_DATASET_NAME}}"
OUTPUT_DATASET_NAME="${2:-}"
OVERWRITE_TCP="${3:-false}"

HF_USER=$(
  hf auth whoami \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)

if [[ -z "${HF_USER}" ]]; then
  echo "Failed to determine HF_USER from 'hf auth whoami'." >&2
  exit 1
fi

if [[ "${DATASET_NAME}" == */* ]]; then
  SOURCE_REPO_ID="${DATASET_NAME}"
  SOURCE_DATASET_BASENAME="${DATASET_NAME##*/}"
else
  SOURCE_REPO_ID="${HF_USER}/${DATASET_NAME}"
  SOURCE_DATASET_BASENAME="${DATASET_NAME}"
fi

if [[ -n "${OUTPUT_DATASET_NAME}" ]]; then
  if [[ "${OUTPUT_DATASET_NAME}" == */* ]]; then
    DEST_REPO_ID="${OUTPUT_DATASET_NAME}"
  else
    DEST_REPO_ID="${HF_USER}/${OUTPUT_DATASET_NAME}"
  fi
else
  DEST_REPO_ID="${HF_USER}/${SOURCE_DATASET_BASENAME}${DEFAULT_OUTPUT_SUFFIX}"
fi

CMD=(
  uv run --extra romoya python convert_dataset_add_tcp_action.py
  --repo-id "${SOURCE_REPO_ID}"
  --new-repo-id "${DEST_REPO_ID}"
)

if [[ "${OVERWRITE_TCP}" == "true" ]]; then
  CMD+=(--overwrite-tcp)
fi

"${CMD[@]}"
