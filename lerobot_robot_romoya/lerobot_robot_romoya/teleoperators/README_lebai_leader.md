# Romoya Lebai Leader Teleop

Teleoperator type: `romoya_lebai_leader`

Arm motion source:
- Leader Lebai arm joint positions

Additional keyboard-controlled action fields:
- `gripper.pos`
- `DO_0`
- `DO_1`

Key mapping:
- `Right Arrow`: `gripper.pos = 99`
- `Left Arrow`: `gripper.pos = 0`
- `1`: `DO_0 = 1`, `DO_1 = 1`
- `2`: `DO_0 = 0`, `DO_1 = 1`
- `3`: `DO_0 = 0`, `DO_1 = 0`
- `Esc`: disconnect teleoperator

Notes:
- Joint actions come directly from the leader arm's current joint positions.
- By default, the teleoperator enters Lebai teach mode on connect.
