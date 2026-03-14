uv run --extra romoya python - <<'PY'
from pathlib import Path

from lerobot_robot_romoya.policies.configuration_act_romoya import ACTRomoyaConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyProcessorPipeline

repo_id = "HF_USER_NAME/REPO-ID"
model_dir = Path("outputs/train/EXP-NAME/checkpoints/last/pretrained_model")

config = PreTrainedConfig.from_pretrained(model_dir)
policy_cls = get_policy_class(config.type)
policy = policy_cls.from_pretrained(model_dir)
policy.push_to_hub(repo_id)

preprocessor = PolicyProcessorPipeline.from_pretrained(
    model_dir,
    config_filename="policy_preprocessor.json",
)
preprocessor.push_to_hub(repo_id)

postprocessor = PolicyProcessorPipeline.from_pretrained(
    model_dir,
    config_filename="policy_postprocessor.json",
)
postprocessor.push_to_hub(repo_id)
PY
