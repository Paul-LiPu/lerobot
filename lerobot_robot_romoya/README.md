# lerobot_robot_romoya

Romoya LeRobot plugin package.

Current devices:

- `romoya_lebai_follower`: Lebai follower robot arm
- `romoya_lebai_tcp_follower`: Lebai follower with translational TCP delta actions
- `romoya_lebai_delta_tcp_follower`: Lebai follower with translational and rotational TCP delta actions
- `romoya_lebai_leader`: Lebai leader teleoperator arm
- `romoya_lebai_keyboard`: Keyboard teleoperator for Lebai TCP control

Tested dependency versions:

- `lebai-sdk==0.3.7`
- `lerobot==0.5.1`

The Lebai Python SDK is treated as an external runtime prerequisite. Install it in
the same environment yourself before using the Lebai robots or teleoperators.
This plugin now uses the legacy `lebai_sdk` path, matching the behavior in your
Cooking Robot code.

## Option 1: Install This Plugin Separately

Install the plugin in editable mode:

```bash
cd lerobot_robot_romoya
pip install -e .
pip install lebai-sdk==0.3.7
```

Then run LeRobot normally:

```bash
lerobot-teleoperate \
  --robot.type=romoya_lebai_follower \
  --robot.ip=10.20.17.1 \
  --robot.simu=false \
  --teleop.type=romoya_lebai_leader \
  --teleop.ip=0.0.0.0
```

## Option 2: Run Directly From This Forked LeRobot Repo

This forked LeRobot repo already includes a local `uv` source mapping for this plugin in its root `pyproject.toml`.

Run with the `romoya` extra enabled:

```bash
uv run --extra romoya lerobot-teleoperate \
  --robot.type=romoya_lebai_follower \
  --robot.ip=10.20.17.1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=romoya_lebai_leader \
  --teleop.ip=0.0.0.0 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true
```

Keyboard TCP teleoperation from this forked repo:

```bash
uv run --extra romoya lerobot-teleoperate \
  --robot.type=romoya_lebai_tcp_follower \
  --robot.ip=10.20.17.1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=romoya_lebai_keyboard \
  --teleop.id=my_awesome_keyboard \
  --display_data=true
```

Keyboard full delta-TCP teleoperation from this forked repo:

```bash
uv run --extra romoya lerobot-teleoperate \
  --robot.type=romoya_lebai_delta_tcp_follower \
  --robot.ip=10.20.17.1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=romoya_lebai_keyboard \
  --teleop.id=my_awesome_keyboard \
  --teleop.translation_step_m=0.01 \
  --teleop.rotation_step_rad=0.1 \
  --teleop.gripper_step=10.0 \
  --display_data=true
```

## Teleoperation Notes

The current `romoya_lebai_leader` teleoperator uses:

- the Lebai leader arm for arm joint teleoperation
- keyboard input for end-effector control

Keyboard bindings:

- `Right Arrow`: open gripper to `99`
- `Left Arrow`: close gripper to `0`
- `1`: `DO_0=1`, `DO_1=1`
- `2`: `DO_0=0`, `DO_1=1`
- `3`: `DO_0=0`, `DO_1=0`
- `Esc`: disconnect teleoperator

The `romoya_lebai_keyboard` teleoperator uses:

- `Up Arrow`: `delta_y = -translation_step_m`
- `Down Arrow`: `delta_y = translation_step_m`
- `Left Arrow`: `delta_x = translation_step_m`
- `Right Arrow`: `delta_x = -translation_step_m`
- `Left Shift`: `delta_z = -translation_step_m`
- `Right Shift`: `delta_z = translation_step_m`
- `W`: `delta_wx = rotation_step_rad`
- `S`: `delta_wx = -rotation_step_rad`
- `A`: `delta_wy = rotation_step_rad`
- `D`: `delta_wy = -rotation_step_rad`
- `Q`: `delta_wz = rotation_step_rad`
- `E`: `delta_wz = -rotation_step_rad`
- `Right Ctrl`: open gripper by `gripper_step`
- `Left Ctrl`: close gripper by `gripper_step`
- `1`: `DO_0=1`, `DO_1=1`
- `2`: `DO_0=0`, `DO_1=1`
- `3`: `DO_0=0`, `DO_1=0`
- `Esc`: disconnect teleoperator
