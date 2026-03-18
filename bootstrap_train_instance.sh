#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh cloud instance for Romoya LeRobot training.
#
# One-line download and run:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Paul-LiPu/lerobot/customized/bootstrap_train_instance.sh)"
#
# Optional overrides:
#   REPO_DIR=/workspace/lerobot BRANCH=customized bash bootstrap_train_instance.sh

REPO_URL="${REPO_URL:-https://github.com/Paul-LiPu/lerobot.git}"
REPO_DIR="${REPO_DIR:-/workspace/lerobot}"
BRANCH="${BRANCH:-customized}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${REPO_DIR}/train_act_side_config.json}"

ensure_cmd() {
  local cmd="$1"
  local install_hint="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    echo "${install_hint}" >&2
    exit 1
  fi
}

echo "==> Checking required commands"
ensure_cmd git "Install git first."
ensure_cmd python3 "Install python3 first."
ensure_cmd pip "Install pip first."
ensure_cmd uv "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"

echo "==> Installing CLI tools"
python3 -m pip install --upgrade "huggingface_hub[cli]" wandb
ensure_cmd hf "Install the Hugging Face CLI with: python3 -m pip install --upgrade 'huggingface_hub[cli]'"
ensure_cmd wandb "Install Weights & Biases with: python3 -m pip install --upgrade wandb"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "==> Cloning repo to ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  echo "==> Repo already exists at ${REPO_DIR}, skipping clone"
fi

cd "${REPO_DIR}"

echo "==> Fetching latest refs"
git fetch origin

echo "==> Checking out branch ${BRANCH}"
git checkout "${BRANCH}"

echo "==> Syncing project environment"
uv sync --extra romoya

echo "==> Checking Hugging Face login"
HF_USER=$(
  hf auth whoami 2>/dev/null \
    | python3 -c 'import re,sys; s=sys.stdin.read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s); print(s, end="")' \
    | awk -F': *' 'NR==1 {print $2}'
)
if [[ -z "${HF_USER}" ]]; then
  echo "Hugging Face login required. Running: hf auth login"
  hf auth login
fi

echo "==> Checking Weights & Biases login"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}"
elif ! grep -q 'machine api\.wandb\.ai' "${HOME}/.netrc" 2>/dev/null; then
  echo "Weights & Biases login required."
  wandb login
fi

echo "==> Ready to train"
echo "Run:"
echo "  cd ${REPO_DIR}"
echo "  TRAIN_CONFIG_PATH=${TRAIN_CONFIG_PATH} bash train.sh"
