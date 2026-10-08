# Real-robot test: convention alignment and procedure

The goal was zero-shot transfer: the checkpoint trained in simulation runs on the robot with
nothing added at evaluation time, no normalisation, no action filtering, no extra inputs, no
threshold changes. The conventions therefore have to agree at training and data-writing
time, not be patched at deployment. This page records the alignment checks and how the
test is run. Credentials, machine names and lab network details are not in this repository;
the launcher takes them from the environment.

## Convention alignment (checked against source, 2026-08-13)

**Action space.** Labels are seven absolute joint positions plus a binary gripper.

- Training: openpi `pi05_droid_jointpos_polaris` has `action_space=JOINT_POSITION`;
  `RLDSDroidDataConfig` pushes `DeltaActions(make_bool_mask(7,-1))` itself, so the dataset
  holds absolute angles and the server also emits absolute angles; the gripper dimension is
  absolute throughout.
- Simulation evaluation: PolaRiS `droid_cfg.py` uses `JointPositionActionCfg(use_default_offset=False)`
  and `BinaryJointPositionZeroToOne` (> 0.5 closes).
- Robot: DROID `robot.py:240`, with `action_space="joint_position"`,
  `action_dict["joint_position"] = action[:-1]` passes absolute radians straight to
  `update_desired_joint_positions`. `robot_env.py:24`'s `check_action_range` applies only to
  the velocity space, so the position path has no normalisation and no range contract.

**Control rate.** 15 Hz on all three sides (sim: decimation 8 × dt 1/120; robot:
`robot_ik_solver.py` `control_hz = 15`).

**Observation packing.** PolaRiS's `droid_jointpos_client.py` and DROID's `evaluate_openpi.py`
send the same keys through the same function:

| key | sim | robot |
|---|---|---|
| `observation/exterior_image_1_left` | `resize_with_pad(external_cam, 224, 224)` | `resize_with_pad(left camera, 224, 224)` |
| `observation/wrist_image_left` | `resize_with_pad(wrist_cam, 224, 224)` | `resize_with_pad(wrist camera, 224, 224)` |
| `observation/joint_position` | measured qpos (7) | `robot_state["joint_positions"]` |
| `observation/gripper_position` | `finger_joint / (π/4)`, clipped to 0..1 | `1 - width / max_width` |
| `prompt` | the instruction in `initial_conditions.json` | typed instruction |

Three details: the image resize is the same function on both sides (openpi's training
transform `ResizeImages(224,224)` is `image_tools.resize_with_pad`; the 180×320 training
frames and the 1280×720 evaluation frames are both 16:9, so the padding geometry matches);
the gripper direction agrees (0 open, 1 closed on both sides); both images are RGB on the
robot (`[..., :3]` then `[..., ::-1]` on both streams).

**Commanded speeds are within limits.** Evaluation videos run 8x fast (one stored frame per
inference, open-loop horizon 8), so the commanded rate was measured directly with
`scripts/eval/log_action_rates.py` on the final checkpoint over two successful episodes
(2100 steps):

| | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| peak rad/s | 0.81 | 1.20 | 1.06 | 0.91 | 1.10 | 0.54 | 0.90 |
| fraction of the wall | 0.51 | 0.76 | 0.67 | 0.57 | 0.55 | 0.27 | 0.45 |

2098 velocity samples, overall peak 0.76 of the ceiling, zero over-limit steps. The training
data's median peak is 0.77, so the policy learned the motion budget rather than speeding up.
The gripper output in the same recording is exactly {0.0, 1.0}, so binarising on the robot is
lossless. Episodes must be cut apart before differencing: the step across an episode boundary
is a reset to home and would read as a 19x overspeed.

**Identical start pose.** DROID's `reset_joints` is `[0, -π/5, 0, -4π/5, 0, 3π/5, 0]`; the
182 training trajectories start within 1e-7 rad of it per joint.

**One exterior camera.** `droid_policy.py:44-55` reads only `exterior_image_1_left` and
`wrist_image_left`; the second exterior slot is zeros. Writing the same frame into both
exterior fields at data time is therefore harmless, but the physical camera mapped to
`exterior_image_1_left` must be the one the training viewpoint models.

**Norm stats.** The official ones, never recomputed. The server log must show
`Loaded norm stats from gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets/droid`;
the checkpoint's own copy is byte-identical (md5 `385156b4295b2efd7c342d5063bce321`). If the
log points at a self-computed directory, stop.

**Instruction verbatim.** `pour the mustard into the blue cup`, exactly as in the data.

## Required client changes

DROID's `evaluate_openpi.py` is written for the joint-velocity head and would wreck a
position-head rollout. `scripts/real_robot/run_eval_safe.py` patches its source in memory and
executes it; the file on disk is never modified (that repository is shared with other
projects). Each patch must match exactly once or the launcher aborts, so an upstream edit can
never silently skip a change. `--show-diff` prints the patched script without touching the
robot. The patches, with the snapshot's line numbers:

1. Line 79: `RobotEnv(action_space="joint_velocity")` → `joint_position`. `check_action_range`
   then turns off, the ±1 assertion no longer fires, and the DoF count goes from 7 to 8.
2. Line 163: the unconditional `action = np.clip(action, -1, 1)` is removed for the arm
   dimensions. Absolute joint angles clipped to ±1 rad destroy the trajectory (j4 normally
   sits near -2.5), and nothing downstream would report it.
3. Line 41: `max_timesteps` 600 → 1050 (70 s at 15 Hz). Training episodes are 348 steps at
   the median, 665 at the 90th percentile, 969 at most; 600 would cut the long ones before the
   pour. The simulation evaluation uses 1050 as well.
4. Lines 155-160: the gripper binarisation (> 0.5) is re-enabled, because the sim client
   binarises (`droid_jointpos_client.py:87`) and the data's gripper column is strictly {0, 1}.
   Without it a 0.87 output closes fully in sim but to 87% on the robot.

Three more were added after the first session (2026-08-14), all about gripper timing. The
symptom was: grasped the bottle but neither lifted nor poured, the arm dithering near the
grasp until timeout.

5. Hold position after commanding close until the gripper reading arrives (up to 28 steps).
   In sim the gripper is a binary joint and reads closed on the next step (median 1 step in
   the training set); the real 2F-85's full 0.085 m stroke at `goto(speed=0.05)` takes about
   1.7 s, about 25 steps. During those steps the policy sees "close commanded, still reads
   open", a state that never occurs in the 182 training trajectories, and stays in the grasp
   phase; with `open_loop_horizon=8` it re-infers three times and reaches the same conclusion.
   Repeating the same absolute joint target holds the arm until the fingers arrive, after
   which the observation matches the training set's "grasped" frame. This is the one genuine
   dynamics gap found; with it, ep09/ep11/ep12/ep13 succeeded.
6. Lock the gripper once it holds the bottle. In the 216 training trajectories the gripper
   switches once (0 → 1) and never back, so reopening after a grasp is out of distribution;
   the first session saw four open/close flutters after a grasp. The lock keys on the measured
   reading, not the command: holding the bottle reads 0.30-0.42 (five measurements; sim
   0.376), a missed grasp closes to about 0.85, and only the former locks, so a miss can retry.
7. The scoring script recorded `y` as 1% (assigned 1.0 and then divided by 100 with the
   numeric path); changed to 100.0.

Patches 5 and 6 are compensations, not fixes. The fix belongs on the generation side: leave
enough steps for the gripper to close (25 steps or the measured stroke time) so the policy
learns to wait, after which patch 5 can go. Patch 6 matches the data (no release in the
training set) and can stay.

The launcher also runs a motion watchdog by default (`--no-guard` disables it). It judges and
never modifies: when it does not trigger the execution is identical to running without it.
It aborts on an absolute joint target outside the FR3-and-Panda intersection
(`docs/fr3_limits.md`) or on a step-to-step displacement that implies a velocity above the
soft wall the data was generated under (1.575 rad/s J1-4, 2.01 J5-7). On the final checkpoint
the measured peak is 0.76 of that wall, so a trigger means something is wrong.

## Before touching the robot

1. Re-read the execution-layer limits on the robot's controller (`config/fr3/franka_hardware.yaml`).
   The generator's soft wall (1.575 / 2.01 rad/s; hard 2.075 / 2.51) is a transcription; if
   the file disagrees, regenerate the data.
2. Confirm which physical camera is `exterior_image_1_left` (`--left_camera_id`). Training's
   exterior view is the deployment-position ZED 2i. The wrong camera is a different viewpoint.
3. Place the objects inside the training distribution, which is narrow. Measured on the 182
   training episodes (`scripts/datagen/analyse_dataset.py`, base frame, metres):

   | object | x | y |
   |---|---|---|
   | mustard bottle | 0.403 - 0.501 | -0.144 - -0.050 |
   | blue cup | 0.442 - 0.501 | 0.126 - 0.184 |

   The bottle appears in a 10×9 cm box with yaw ±10 degrees, the cup in 6×6 cm with yaw ±15
   degrees, the cup always 0.19-0.32 m to the bottle's +y side. Outside those boxes a failure
   is not the model's. The range follows R2R2R's (±6 cm) and the lab's earlier recipe (±5 cm);
   widening it requires retargeting the pour, not just changing a sampling bound.
   `scripts/real_robot/check_placement.py` reads the bottle's apparent size in the wrist view
   and says whether the placement matches the sim median (every 2026-08-14 failure measured
   0.95-1.06% of the wrist frame against sim's 1.31%; every success 1.14-1.37%).
4. After starting the server check three lines: norm stats from the official assets,
   `server listening on 0.0.0.0:8000`, and `Predicted action chunk shape: (15, 8)` on the first
   step. Both norm-stats lines (the config's official assets and the checkpoint's own copy)
   must appear and are identical.
5. Only the left arm is used; a hand stays on the emergency stop for the first episode.

## Running

On the server, with this repository on `PYTHONPATH` (the training config registers only when
`r2s2r` is importable; otherwise openpi reports `Config ... not found`):

```bash
cd "$POLARIS_ROOT/third_party/openpi"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
PYTHONPATH=<this repo>/src \
  uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_droid_jointpos_polaris_pourmustard \
  --policy.dir <checkpoints>/pi05_droid_jointpos_polaris_pourmustard/<run>/9999
```

The final checkpoint directory is `9999`, not `10000`: openpi counts from 0 and the step-10k
save is written as 9999. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` is enough (13.8 GB measured)
and leaves room for a simulator on the same GPU.

On the robot laptop, with the SSH tunnel up and the camera docker container stopped:

```bash
export DROID_ROOT=<DROID client repo> NUC_USER=<nuc login> NUC_CONTAINER=<nuc docker container> \
       NUC_SUDO_PASSWORD=<...>
python <this repo>/scripts/real_robot/run_eval_safe.py --remote_port 8000 --model pi05
```

Do not run DROID's `evaluate_openpi.py` directly; none of the patches above would apply. The
launcher also works around a gripper start-up race on the NUC (`launch_gripper.sh` runs
`pkill -9` and the restart with no delay; a Robotiq client killed mid-Modbus reports
`Unable to activate!`, which surfaces later as
`'GripperInterface' object has no attribute 'metadata'`); `--check-only` brings the stack up
and verifies it without running an episode. `open_loop_horizon` stays at 8 (same as the sim
evaluation; the model's `action_horizon` is 15).

## Troubleshooting

- Barely moves: check the two norm-stats lines, then whether `clip(-1, 1)` is really gone.
- Too fast, swings towards the table: emergency stop. On the position head this usually means
  the action was interpreted in another convention; check that `action_space` really is
  `joint_position`.
- Grasps but does not lift or pour: gripper stroke time, patch 5. If the gripper or its
  closing speed changes, `GRIPPER_CLOSE_STEPS` changes with it.
- Does not hold the bottle: patches 5 and 6, then compare the same trajectory's replay in sim.
- Cameras do not open: the laptop's docker container was not stopped.
- Wrong viewpoint: `left_camera_id` is not the deployment-position camera.

The 2026-08-14 session and its outcome per episode are in `results/real_robot_20260814/`.
