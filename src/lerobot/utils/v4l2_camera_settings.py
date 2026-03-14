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
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTROL_LINE_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+(0x[0-9a-fA-F]+)\s+\(([^)]+)\)\s*:\s*(.*)$")
MENU_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")
ATTRIBUTE_RE = re.compile(r"(\w+)=([^=]+?)(?=\s+\w+=|$)")
VIDEO_DEVICE_RE = re.compile(r"^\s*(/dev/video\d+)\s*$")
DEVICE_CAPS_RE = re.compile(r"^\s*Device Caps\s*:\s*(0x[0-9a-fA-F]+)\s*$")
CAPABILITIES_RE = re.compile(r"^\s*Capabilities\s*:\s*(0x[0-9a-fA-F]+)\s*$")
KEY_VALUE_LINE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")
WIDTH_HEIGHT_RE = re.compile(r"Width/Height\s*:\s*(\d+)/(\d+)")
PIXEL_FORMAT_RE = re.compile(r"Pixel Format\s*:\s*'([^']+)'")
FPS_RE = re.compile(r"Frames per second\s*:\s*([0-9]+(?:\.[0-9]+)?)")
FRAME_INTERVAL_RE = re.compile(r"Frame period\s*:\s*([0-9.]+)s")


@dataclass
class V4L2MenuItem:
    index: int
    label: str


@dataclass
class V4L2Control:
    name: str
    control_id: str
    control_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    menu_items: list[V4L2MenuItem] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        flags = self.attributes.get("flags", [])
        return flags if isinstance(flags, list) else []

    @property
    def value(self) -> Any:
        return self.attributes.get("value")


def _convert_attribute_value(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _parse_attributes(text: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, raw_value in ATTRIBUTE_RE.findall(text):
        value = _convert_attribute_value(raw_value)
        if key == "flags":
            value = [flag.strip() for flag in str(value).split(",") if flag.strip()]
        attributes[key] = value
    return attributes


def parse_v4l2_control_listing(text: str) -> list[V4L2Control]:
    controls: list[V4L2Control] = []
    current_control: V4L2Control | None = None

    for line in text.splitlines():
        if not line.strip():
            continue

        control_match = CONTROL_LINE_RE.match(line)
        if control_match:
            current_control = V4L2Control(
                name=control_match.group(1),
                control_id=control_match.group(2),
                control_type=control_match.group(3),
                attributes=_parse_attributes(control_match.group(4)),
            )
            controls.append(current_control)
            continue

        menu_match = MENU_LINE_RE.match(line)
        if menu_match and current_control is not None and current_control.control_type == "menu":
            current_control.menu_items.append(
                V4L2MenuItem(index=int(menu_match.group(1)), label=menu_match.group(2))
            )

    return controls


def is_control_restorable(control: V4L2Control) -> bool:
    if control.control_type in {"button", "class"}:
        return False
    if control.value is None:
        return False
    flags = set(control.flags)
    if "read-only" in flags:
        return False
    return True


def _run_v4l2_ctl(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["v4l2-ctl", *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`v4l2-ctl` is required. Install `v4l-utils` first.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(stderr or f"`v4l2-ctl {' '.join(args)}` failed.") from exc

    return completed.stdout


def parse_v4l2_device_listing(text: str) -> dict[str, list[str]]:
    devices: dict[str, list[str]] = {}
    current_name: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            current_name = None
            continue

        if not raw_line[0].isspace():
            current_name = line.rstrip(":")
            devices[current_name] = []
            continue

        match = VIDEO_DEVICE_RE.match(line)
        if match and current_name is not None:
            devices[current_name].append(match.group(1))

    return {name: paths for name, paths in devices.items() if paths}


def list_video_devices() -> dict[str, list[str]]:
    output = _run_v4l2_ctl("--list-devices")
    return parse_v4l2_device_listing(output)


def parse_v4l2_capabilities(text: str) -> list[str]:
    device_capabilities: list[str] = []
    generic_capabilities: list[str] = []
    current_target: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if DEVICE_CAPS_RE.match(line):
            current_target = device_capabilities
            continue

        if CAPABILITIES_RE.match(line):
            current_target = generic_capabilities
            continue

        if current_target is not None:
            if not raw_line.startswith("\t") and not raw_line.startswith(" "):
                current_target = None
                continue
            stripped = line.strip()
            if stripped:
                current_target.append(stripped)

    return device_capabilities or generic_capabilities


def is_video_capture_device(device: str) -> bool:
    output = _run_v4l2_ctl("--device", device, "--all")
    capabilities = parse_v4l2_capabilities(output)
    return "Video Capture" in capabilities


def parse_v4l2_all_key_values(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = KEY_VALUE_LINE_RE.match(line.rstrip())
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            fields[key] = value
    return fields


def get_video_device_identity(device: str) -> str:
    output = _run_v4l2_ctl("--device", device, "--all")
    fields = parse_v4l2_all_key_values(output)
    return fields.get("Bus info") or fields.get("Serial") or fields.get("Card type") or device


def get_stream_config(device: str) -> dict[str, Any]:
    fmt_output = _run_v4l2_ctl("--device", device, "--get-fmt-video")
    parm_output = _run_v4l2_ctl("--device", device, "--get-parm")

    stream: dict[str, Any] = {}

    width_height = WIDTH_HEIGHT_RE.search(fmt_output)
    if width_height:
        stream["width"] = int(width_height.group(1))
        stream["height"] = int(width_height.group(2))

    pixel_format = PIXEL_FORMAT_RE.search(fmt_output)
    if pixel_format:
        stream["pixel_format"] = pixel_format.group(1)

    fps_match = FPS_RE.search(parm_output)
    if fps_match:
        stream["fps"] = float(fps_match.group(1))
    else:
        frame_interval = FRAME_INTERVAL_RE.search(parm_output)
        if frame_interval:
            period_s = float(frame_interval.group(1))
            if period_s > 0:
                stream["fps"] = 1.0 / period_s

    return stream


def _stream_preference_key(stream: dict[str, Any]) -> tuple[int, int, float]:
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps = float(stream.get("fps", 0.0))
    return (1 if width >= 1080 else 0, width * height, fps)


def list_controls(device: str) -> list[V4L2Control]:
    output = _run_v4l2_ctl("--device", device, "--list-ctrls-menus")
    return parse_v4l2_control_listing(output)


def _control_payload(control: V4L2Control) -> dict[str, Any]:
    return {
        "name": control.name,
        "control_id": control.control_id,
        "control_type": control.control_type,
        "attributes": control.attributes,
        "menu_items": [asdict(item) for item in control.menu_items],
    }


def _device_payload(device: str, display_name: str | None = None) -> dict[str, Any]:
    controls = list_controls(device)
    payload: dict[str, Any] = {
        "stream": get_stream_config(device),
        "controls": [_control_payload(control) for control in controls],
    }
    if display_name is not None:
        payload["display_name"] = display_name
    return payload


def save_controls(device: str, output_path: Path) -> dict[str, Any]:
    return save_selected_controls([device], output_path)


def save_selected_controls(devices: list[str], output_path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "devices": {},
    }
    for device in devices:
        payload["devices"][device] = _device_payload(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def save_all_controls(output_path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "devices": {},
    }
    selected_devices: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for display_name, device_paths in list_video_devices().items():
        for device in device_paths:
            if not is_video_capture_device(device):
                continue
            identity = get_video_device_identity(device)
            stream = get_stream_config(device)
            current = selected_devices.get(identity)
            if current is None or _stream_preference_key(stream) > _stream_preference_key(current[2]):
                selected_devices[identity] = (device, display_name, stream)

    for device, display_name, stream in selected_devices.values():
        payload["devices"][device] = _device_payload(device, display_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_controls(input_path: Path) -> dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def _restore_device_controls(
    device: str, device_payload: dict[str, Any], *, ignore_errors: bool = False
) -> tuple[list[str], list[str]]:
    applied: list[str] = []
    skipped: list[str] = []

    stream = device_payload.get("stream", {})
    if stream:
        stream_args: list[str] = ["--device", device]
        if "width" in stream and "height" in stream:
            stream_args.append(f"--set-fmt-video=width={stream['width']},height={stream['height']}")
            if "pixel_format" in stream:
                stream_args[-1] += f",pixelformat={stream['pixel_format']}"
        elif "pixel_format" in stream:
            stream_args.append(f"--set-fmt-video=pixelformat={stream['pixel_format']}")

        if len(stream_args) > 2:
            _run_v4l2_ctl(*stream_args)
            applied.append("__stream_format__")

        if "fps" in stream:
            fps_value = stream["fps"]
            fps_text = str(int(fps_value)) if float(fps_value).is_integer() else str(fps_value)
            _run_v4l2_ctl("--device", device, f"--set-parm={fps_text}")
            applied.append("__stream_fps__")

    for control_data in device_payload.get("controls", []):
        control = V4L2Control(
            name=control_data["name"],
            control_id=control_data["control_id"],
            control_type=control_data["control_type"],
            attributes=control_data.get("attributes", {}),
            menu_items=[V4L2MenuItem(**item) for item in control_data.get("menu_items", [])],
        )

        if not is_control_restorable(control):
            skipped.append(control.name)
            continue

        value = control.value
        try:
            _run_v4l2_ctl("--device", device, f"--set-ctrl={control.name}={value}")
            applied.append(control.name)
        except RuntimeError:
            if ignore_errors:
                skipped.append(control.name)
                continue
            raise

    return applied, skipped


def iter_payload_devices(payload: dict[str, Any], selected_device: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    if "devices" in payload:
        devices = payload["devices"]
        if selected_device is not None:
            if selected_device not in devices:
                raise ValueError(f"Device {selected_device} not found in settings file.")
            return [(selected_device, devices[selected_device])]
        return list(devices.items())

    if "device" in payload and "controls" in payload:
        device = selected_device or payload["device"]
        return [(device, payload)]

    raise ValueError("Unrecognized V4L2 settings file format.")


def restore_controls(
    payload: dict[str, Any],
    *,
    device: str | None = None,
    ignore_errors: bool = False,
) -> dict[str, dict[str, list[str]]]:
    results: dict[str, dict[str, list[str]]] = {}
    for resolved_device, device_payload in iter_payload_devices(payload, selected_device=device):
        applied, skipped = _restore_device_controls(
            resolved_device,
            device_payload,
            ignore_errors=ignore_errors,
        )
        results[resolved_device] = {"applied": applied, "skipped": skipped}
    return results
