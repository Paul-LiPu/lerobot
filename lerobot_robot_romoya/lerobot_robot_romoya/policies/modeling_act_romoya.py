from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_act_romoya import ACTRomoyaConfig


class ACTRomoyaPolicy(ACTPolicy):
    config_class = ACTRomoyaConfig
    name = "act_romoya"
