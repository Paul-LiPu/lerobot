#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Save and restore UVC/V4L2 camera control settings through `v4l2-ctl`.

Example:

```shell
lerobot-v4l2-camera-settings save --device /dev/video0 --output camera-settings.json
lerobot-v4l2-camera-settings load --device /dev/video0 --input camera-settings.json
```
"""

import argparse
from pathlib import Path

from lerobot.utils.v4l2_camera_settings import (
    load_controls,
    restore_controls,
    save_all_controls,
    save_controls,
    save_selected_controls,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save", help="Save current V4L2 control values to JSON.")
    save_parser.add_argument(
        "--device",
        action="append",
        help="V4L2 device path, e.g. /dev/video0. Repeat to save multiple exact device paths. Omit to save all devices.",
    )
    save_parser.add_argument(
        "--output",
        default=Path("camera-settings.json"),
        type=Path,
        help="JSON file to write. Defaults to ./camera-settings.json.",
    )

    load_parser = subparsers.add_parser("load", help="Restore V4L2 control values from JSON.")
    load_parser.add_argument(
        "--device",
        help="V4L2 device path, e.g. /dev/video0. Omit to restore all devices found in the JSON file.",
    )
    load_parser.add_argument(
        "--input",
        default=Path("camera-settings.json"),
        type=Path,
        help="JSON file to read. Defaults to ./camera-settings.json.",
    )
    load_parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Skip controls that fail to restore instead of exiting immediately.",
    )

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "save":
        if not args.device:
            payload = save_all_controls(args.output)
        elif len(args.device) == 1:
            payload = save_controls(args.device[0], args.output)
        else:
            payload = save_selected_controls(args.device, args.output)
        print(f"Saved {len(payload['devices'])} device(s) to {args.output}")
        return

    payload = load_controls(args.input)
    results = restore_controls(payload, device=args.device, ignore_errors=args.ignore_errors)
    for device, result in results.items():
        print(f"{device}: applied {len(result['applied'])} controls")
        if result["skipped"]:
            print(f"{device}: skipped {len(result['skipped'])} controls: {', '.join(result['skipped'])}")


if __name__ == "__main__":
    main()
