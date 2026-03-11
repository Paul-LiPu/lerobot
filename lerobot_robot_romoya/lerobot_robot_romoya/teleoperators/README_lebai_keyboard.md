# Romoya Lebai Keyboard Teleop

Teleoperator type: `romoya_lebai_keyboard`

Action fields:
- `delta_x`
- `delta_y`
- `delta_z`
- `delta_wx`
- `delta_wy`
- `delta_wz`
- `gripper`
- `DO_0`
- `DO_1`

Key mapping:
- `Up Arrow`: `delta_y = -1`
- `Down Arrow`: `delta_y = 1`
- `Left Arrow`: `delta_x = 1`
- `Right Arrow`: `delta_x = -1`
- `Left Shift`: `delta_z = -1`
- `Right Shift`: `delta_z = 1`
- `W`: `delta_wx = 1`
- `S`: `delta_wx = -1`
- `A`: `delta_wy = 1`
- `D`: `delta_wy = -1`
- `Q`: `delta_wz = 1`
- `E`: `delta_wz = -1`
- `Right Ctrl`: `gripper = 2` (open)
- `Left Ctrl`: `gripper = 0` (close)
- no gripper key: `gripper = 1` (hold)
- `1`: `DO_0 = 1`, `DO_1 = 1`
- `2`: `DO_0 = 0`, `DO_1 = 1`
- `3`: `DO_0 = 0`, `DO_1 = 0`
- `Esc`: disconnect teleoperator

Default step sizes are defined by the robot config that consumes these actions:
- translation: `cartesian_step_m`
- rotation: `rotation_step_rad`
- gripper: `gripper_step`
