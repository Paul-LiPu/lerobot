#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    policy = payload.get("policy", payload)
    if policy.get("type") not in {"act_romoya", "pi05_romoya"}:
        print("none")
        return

    action_mode = policy.get("action_mode", "__missing__")
    if action_mode is not None and action_mode != "__missing__":
        print("legacy")
        return

    binary_state = policy.get("binary_state") or []
    binary_action = policy.get("binary_action") or []
    delta_action = policy.get("delta_action") or []
    needs_prepare = any(value is not None for value in binary_state) or any(
        value is not None for value in binary_action
    ) or any(bool(value) for value in delta_action)
    print("generic" if needs_prepare else "none")


if __name__ == "__main__":
    main()
