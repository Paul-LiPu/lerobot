#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash regenerate_model_processors.sh /path/to/checkpoint_or_pretrained_model
#
# This rebuilds policy_preprocessor.json / policy_postprocessor.json and their state files
# from the current source code.
#
# It reuses the existing saved normalization stats from the checkpoint itself.
#
# It is useful when a checkpoint was saved with outdated processor construction logic
# but still has the correct saved normalizer/unnormalizer state files.

CHECKPOINT_PATH="${1:-}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Usage: bash regenerate_model_processors.sh /path/to/checkpoint_or_pretrained_model" >&2
  exit 1
fi

uv run --extra romoya --extra pi python - "${CHECKPOINT_PATH}" <<'PY'
from __future__ import annotations

import glob
import json
import shutil
import sys
import tempfile
from pathlib import Path

import lerobot_robot_romoya  # noqa: F401  Registers Romoya policy/config classes.
from lerobot_robot_romoya.policies.configuration_act_romoya import ACTRomoyaConfig
from lerobot_robot_romoya.policies.configuration_pi05_romoya import PI05RomoyaConfig
from lerobot.processor.migrate_policy_normalization import (
    convert_features_to_policy_features,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from safetensors.torch import load_file

checkpoint_path = Path(sys.argv[1]).expanduser().resolve()

if (checkpoint_path / "config.json").is_file():
    model_dir = checkpoint_path
elif (checkpoint_path / PRETRAINED_MODEL_DIR / "config.json").is_file():
    model_dir = checkpoint_path / PRETRAINED_MODEL_DIR
else:
    raise FileNotFoundError(
        f"Could not find config.json under {checkpoint_path} or {checkpoint_path / PRETRAINED_MODEL_DIR}"
    )


def load_processor_stats(path: Path) -> dict[str, dict[str, object]]:
    flat_state = load_file(str(path))
    nested: dict[str, dict[str, object]] = {}
    for flat_key, tensor in flat_state.items():
        feature_name, stat_name = flat_key.rsplit(".", 1)
        nested.setdefault(feature_name, {})[stat_name] = tensor
    return nested

policy_payload = json.loads((model_dir / "config.json").read_text())
policy_type = policy_payload.pop("type")
policy_payload["input_features"] = convert_features_to_policy_features(policy_payload["input_features"])
policy_payload["output_features"] = convert_features_to_policy_features(policy_payload["output_features"])
if policy_type == "pi05_romoya":
    policy_cfg = PI05RomoyaConfig(**policy_payload)
elif policy_type == "act_romoya":
    policy_cfg = ACTRomoyaConfig(**policy_payload)
else:
    raise ValueError(
        f"Unsupported policy type '{policy_type}'. "
        "This repair script currently supports pi05_romoya and act_romoya checkpoints."
    )

preprocessor_state_files = sorted(model_dir.glob("policy_preprocessor_step_*.safetensors"))
postprocessor_state_files = sorted(model_dir.glob("policy_postprocessor_step_*.safetensors"))

if preprocessor_state_files:
    dataset_stats = load_processor_stats(preprocessor_state_files[0])
elif postprocessor_state_files:
    dataset_stats = load_processor_stats(postprocessor_state_files[0])
else:
    raise FileNotFoundError(
        f"Could not find saved processor state files in {model_dir}. "
        "Expected policy_preprocessor_step_*.safetensors or policy_postprocessor_step_*.safetensors."
    )

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg,
    dataset_stats=dataset_stats,
)

with tempfile.TemporaryDirectory(prefix="romoya_processor_regen_") as tmp_dir_str:
    tmp_dir = Path(tmp_dir_str)
    preprocessor.save_pretrained(tmp_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_dir, config_filename="policy_postprocessor.json")

    generated_state_files = sorted(tmp_dir.glob("policy_*_step_*.safetensors"))
    if not generated_state_files:
        raise RuntimeError(
            "Regenerated processors did not emit any processor state files. "
            "Refusing to replace the existing checkpoint artifacts."
        )

    for pattern in (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_*.safetensors",
        "policy_postprocessor_step_*.safetensors",
    ):
        for path in glob.glob(str(model_dir / pattern)):
            Path(path).unlink()

    for path in tmp_dir.iterdir():
        shutil.move(str(path), str(model_dir / path.name))

print(f"Regenerated processors in {model_dir}")
print("Reused normalization stats from existing checkpoint processor state files")
PY
