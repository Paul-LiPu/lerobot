#!/usr/bin/env python

from lerobot.utils.v4l2_camera_settings import (
    get_video_device_identity,
    get_stream_config,
    is_control_restorable,
    iter_payload_devices,
    parse_v4l2_all_key_values,
    parse_v4l2_capabilities,
    parse_v4l2_control_listing,
    parse_v4l2_device_listing,
)


SAMPLE_CONTROLS = """
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=12
                       contrast 0x00980901 (int)    : min=0 max=95 step=1 default=32 value=40
         white_balance_automatic 0x0098090c (bool)   : default=1 value=0
                  power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=2
                        0: Disabled
                        1: 50 Hz
                        2: 60 Hz
                     camera_class 0x009a0001 (class)  : name=Camera Controls
                  exposure_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=125 flags=inactive
                        pan_reset 0x009a0904 (button) : flags=write-only,execute-on-write
"""


SAMPLE_DEVICES = """
HD USB Camera: HD USB Camera (usb-0000:00:14.0-7):
\t/dev/video0
\t/dev/video1

Integrated Camera: Integrated C:
\t/dev/video2
"""


SAMPLE_CAPABILITIES = """
Driver Info:
\tDriver name      : uvcvideo
\tCard type        : HD USB Camera
\tBus info         : usb-0000:00:14.0-7
\tDriver version   : 6.8.0
\tCapabilities     : 0x84A00001
\t\tVideo Capture
\t\tMetadata Capture
\t\tStreaming
\t\tExtended Pix Format
\t\tDevice Capabilities
\tDevice Caps      : 0x04200001
\t\tVideo Capture
\t\tStreaming
\t\tExtended Pix Format
"""


SAMPLE_V4L2_ALL = """
Driver Info:
\tDriver name      : uvcvideo
\tCard type        : Logitech BRIO
\tBus info         : usb-0000:00:14.0-3
\tDriver version   : 6.8.0
\tCapabilities     : 0x84A00001
\t\tVideo Capture
\t\tMetadata Capture
\t\tStreaming
\t\tExtended Pix Format
\t\tDevice Capabilities
\tDevice Caps      : 0x04200001
\t\tVideo Capture
\t\tStreaming
\t\tExtended Pix Format
"""


SAMPLE_FMT_VIDEO = """
Format Video Capture:
\tWidth/Height      : 1920/1080
\tPixel Format      : 'MJPG' (Motion-JPEG, compressed)
\tField             : None
\tBytes per Line    : 0
\tSize Image        : 4147200
\tColorspace        : sRGB
"""


SAMPLE_GET_PARM = """
Streaming Parameters Video Capture:
\tCapabilities     : timeperframe
\tFrames per second: 30.000 (30/1)
\tRead buffers     : 0
"""


def test_parse_v4l2_control_listing():
    controls = parse_v4l2_control_listing(SAMPLE_CONTROLS)

    assert [control.name for control in controls] == [
        "brightness",
        "contrast",
        "white_balance_automatic",
        "power_line_frequency",
        "camera_class",
        "exposure_absolute",
        "pan_reset",
    ]
    assert controls[0].attributes["value"] == 12
    assert controls[3].control_type == "menu"
    assert controls[3].menu_items[2].label == "60 Hz"
    assert controls[4].attributes["name"] == "Camera Controls"
    assert controls[5].attributes["flags"] == ["inactive"]
    assert controls[6].attributes["flags"] == ["write-only", "execute-on-write"]


def test_is_control_restorable():
    controls = parse_v4l2_control_listing(SAMPLE_CONTROLS)

    assert is_control_restorable(controls[0]) is True
    assert is_control_restorable(controls[3]) is True
    assert is_control_restorable(controls[4]) is False
    assert is_control_restorable(controls[6]) is False


def test_parse_v4l2_device_listing():
    devices = parse_v4l2_device_listing(SAMPLE_DEVICES)

    assert devices == {
        "HD USB Camera: HD USB Camera (usb-0000:00:14.0-7)": ["/dev/video0", "/dev/video1"],
        "Integrated Camera: Integrated C": ["/dev/video2"],
    }


def test_iter_payload_devices_schema_v2():
    payload = {
        "schema_version": 2,
        "devices": {
            "/dev/video0": {"controls": []},
            "/dev/video2": {"controls": []},
        },
    }

    assert iter_payload_devices(payload) == [
        ("/dev/video0", {"controls": []}),
        ("/dev/video2", {"controls": []}),
    ]
    assert iter_payload_devices(payload, selected_device="/dev/video2") == [("/dev/video2", {"controls": []})]


def test_parse_v4l2_capabilities():
    capabilities = parse_v4l2_capabilities(SAMPLE_CAPABILITIES)

    assert "Video Capture" in capabilities
    assert "Streaming" in capabilities
    assert "Metadata Capture" in capabilities


def test_parse_v4l2_all_key_values():
    fields = parse_v4l2_all_key_values(SAMPLE_V4L2_ALL)

    assert fields["Driver name"] == "uvcvideo"
    assert fields["Card type"] == "Logitech BRIO"
    assert fields["Bus info"] == "usb-0000:00:14.0-3"


def test_stream_regexes():
    import lerobot.utils.v4l2_camera_settings as mod

    width_height = mod.WIDTH_HEIGHT_RE.search(SAMPLE_FMT_VIDEO)
    pixel_format = mod.PIXEL_FORMAT_RE.search(SAMPLE_FMT_VIDEO)
    fps_match = mod.FPS_RE.search(SAMPLE_GET_PARM)

    assert width_height is not None
    assert width_height.group(1) == "1920"
    assert width_height.group(2) == "1080"
    assert pixel_format is not None
    assert pixel_format.group(1) == "MJPG"
    assert fps_match is not None
    assert fps_match.group(1) == "30.000"
