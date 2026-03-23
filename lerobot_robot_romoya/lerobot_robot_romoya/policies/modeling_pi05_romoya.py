from __future__ import annotations

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from .configuration_pi05_romoya import PI05RomoyaConfig


class PI05RomoyaPolicy(PI05Policy):
    config_class = PI05RomoyaConfig
    name = "pi05_romoya"


__all__ = ["PI05RomoyaPolicy"]
