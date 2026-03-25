#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    dataset_cfg = payload.get("dataset") or {}
    image_transforms = dataset_cfg.get("image_transforms") or {}
    pre_tfs = image_transforms.get("pre_tfs") or {}
    resize_cfg = pre_tfs.get("resize") or {}
    kwargs = resize_cfg.get("kwargs") or {}
    size = kwargs.get("size")
    if isinstance(size, list) and len(size) == 2:
        print(f"--resize-height={int(size[0])}")
        print(f"--resize-width={int(size[1])}")


if __name__ == "__main__":
    main()
