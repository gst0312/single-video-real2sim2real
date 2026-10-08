# FR3 limits: the numbers the generator gates on

Any synthesised trajectory that leaves the ranges below is not generated. Every number here
was read from a primary source on 2026-08-10; none comes from earlier lab notes.

Four sources: the official `franka_description` repository
(github.com/frankarobotics/franka_description, commit 02afaae, version 2.8.1), directories
`robots/fr3/` and `robots/fer/`; PolaRiS's robot USD (`PolaRiS-Hub/nvidia_droid/noninstanceable.usd`)
read with the USD library; the runtime values IsaacLab reports from the articulation; and
libfranka's `include/franka/rate_limiting.h`. `fer` is the Franka Emika Robot, i.e. the
original Panda, which is what the DROID platform and PolaRiS's `nvidia_droid` model.

## Two facts first

FR3 and Panda have identical kinematics: `fr3/kinematics.yaml` and `fer/kinematics.yaml` are
byte-for-byte the same (seven joint xyz/rpy plus the joint8 tool origin at z = 0.107), so the
same joint vector gives the same end-effector pose on both and IK does not need to tell them
apart. What differs is `inertials.yaml` (link masses and inertias, e.g. link1 2.9275 kg on the
FR3 against 2.7907 kg on the Panda) and `joint_limits.yaml`.

PolaRiS's `nvidia_droid` carries the Panda's limits, not the FR3's. The USD stores degrees;
converted to radians they match `fer/joint_limits.yaml` entry for entry, and the running
environment reports the same values from `robot.data.joint_pos_limits` with
`soft_joint_pos_limit_factor = 1`, so the soft limits equal the hard limits.

## Joint position limits (rad)

```
        FR3                          Panda (what PolaRiS runs)
j1      -2.9007  ..  2.9007          -2.8973  ..  2.8973
j2      -1.8361  ..  1.8361          -1.7628  ..  1.7628
j3      -2.9007  ..  2.9007          -2.8973  ..  2.8973
j4      -3.0770  .. -0.1169          -3.0718  .. -0.0698
j5      -2.8763  ..  2.8763          -2.8973  ..  2.8973
j6       0.4398  ..  4.6216          -0.0175  ..  3.7525
j7      -3.0508  ..  3.0508          -2.8973  ..  2.8973
```

FR3 is tighter than Panda in exactly three places, and these are the only positions an FR3
cannot reach that the simulator would accept:

- j4 upper bound is tighter by 0.0471 rad (-0.1169 against -0.0698).
- j5 is tighter by 0.0210 rad on both sides (±2.8763 against ±2.8973).
- j6 lower bound is tighter by 0.4573 rad (0.4398 against -0.0175). This is the one large
  difference: the FR3 cannot go below 0.4398 at all.

On the other four joints FR3 is wider (j2 by 0.0733, j7 by 0.1535, j1/j3 by 0.0034).

Data has to be executable on the FR3 and replayable in the PolaRiS environment, so the
generator gates on the intersection:

```
j1      -2.8973  ..  2.8973
j2      -1.7628  ..  1.7628
j3      -2.8973  ..  2.8973
j4      -3.0718  .. -0.1169
j5      -2.8763  ..  2.8763
j6       0.4398  ..  3.7525
j7      -2.8973  ..  2.8973
```

This array is `polaris_lfhv.limits.LIMITS_LOW / LIMITS_HIGH`; the environment narrows the
simulated arm to the same array at runtime (`PourMustardEnv.FR3_PANDA_LIMITS`). PolaRiS's
reset pose (0, -π/5, 0, -4π/5, 0, 3π/5, 0) = (0, -0.6283, 0, -2.5133, 0, 1.8850, 0) lies well
inside it; j6 at 1.8850 is far from the FR3 floor of 0.4398.

## Joint velocity limits (rad/s)

```
        FR3      Panda spec   PolaRiS config   effective in the sim
j1-j4   2.62     2.175        2.175            10.0
j5      5.26     2.61         2.61             10.0
j6      4.18     2.61         2.61             10.0
j7      5.26     2.61         2.61             10.0
```

Two things to note. FR3's velocity limits are everywhere wider than the Panda's, so velocity
is not where "the FR3 cannot follow" shows up first. And the `velocity_limit` PolaRiS writes
into `robot_cfg.py` for its ImplicitActuator has no effect in IsaacLab 2.3 (it warns
"Previously, although this value was specified, it was not getting used by implicit
actuators"; `velocity_limit_sim` would be needed), so what the simulator enforces is the
USD's 10 rad/s (572.9578 deg/s). The simulator will not stop an over-speed trajectory; the
velocity gate has to live on the generation side.

At the 15 Hz control rate the per-step displacement ceilings are: FR3 0.1747 rad on j1-j4,
0.3507 on j5 and j7, 0.2787 on j6; Panda 0.145 on j1-j4 and 0.174 on j5-j7.

Torque limits are the same on both: 87 N·m on j1-j4, 12 N·m on j5-j7.

## FR3's position-dependent velocity envelope

Near a position limit the FR3 lowers its allowed speed. libfranka's `rate_limiting.h` gives
the formula (its comment cites the "Limits for Franka Research 3" documentation section):
per joint, `min(v_max, max(0, -c + sqrt(max(0, k*(a - q)))))`, with (v_max, c, k, a):

```
j1   2.62  0.30  12.0   2.75010
j2   2.62  0.20   5.17  1.79180
j3   2.62  0.20   7.00  2.90650
j4   2.62  0.30   8.00 -0.14580
j5   5.26  0.35  34.0   2.81010
j6   4.18  0.35  11.0   4.52050
j7   5.26  0.35  34.0   3.01960
```

The negative direction is the mirror image, `max(-v_max, min(0, c - sqrt(max(0, k*(a + q)))))`,
with `a` replaced by 3.0481 for j4 and -0.54092 for j6.

The official documentation ("Control Interface Specification and Robot Limits") writes the
same law with a different parametrisation:

```
dq_max_i(q_i) = min( dq_max_i,  max(0, -dq_offset_i + sqrt(max(0, 2*ddq_dec_i*( q_max_i - q_i)))) )
dq_min_i(q_i) = max( dq_min_i,  min(0,  dq_offset_i - sqrt(max(0, 2*ddq_dec_i*(-q_min_i + q_i)))) )
```

`k = 2*ddq_dec` checks out joint by joint (ddq_dec 6.0, 2.585, 3.5, 4.0, 17.0, 5.5, 17.0
doubles to 12.0, 5.17, 7.00, 8.00, 34.0, 11.0, 34.0). The difference is in the other two
parameters: libfranka's `a` sits further inside than the documented q_max and its `c` is
smaller than the documented offset, so the libfranka envelope is the more conservative one.
For j1 the documented curve reaches zero speed at q = 2.9007 - 0.6599²/12 = 2.864 rad, the
libfranka curve at 2.750 - 0.30²/12 = 2.743 rad, 0.12 rad earlier. libfranka also subtracts a
packet-loss tolerance, which is zero for the FR3. Both are official; the documentation
describes the robot, libfranka the controller with its safety margin. The generator gates on
the libfranka envelope because that is what stops the real arm. libfranka marks these two
functions deprecated from system image 5.9.0, in favour of
`Robot::getUpper/LowerJointVelocityLimits(q)` reported by the firmware; on the robot the
firmware's values are authoritative.

One easy misreading: the documentation's dq_max row can be read as "j6 and j7 are both
4.18"; libfranka's constants are j6 = 4.18, j7 = 5.26.

libfranka's other two constants: joint acceleration limit 10 rad/s², jerk limit 5000 rad/s³.
These are continuous-time limits of the 1 kHz control loop. Differencing a 15 Hz waypoint
stream twice does not measure the same quantity: real teleoperated episodes that the arm has
executed reach 20.77 rad/s² that way (`scripts/datagen/measure_motion_budget.py`), so the
generator's acceleration ceiling is that measured maximum (20.8), and jerk (real maximum
513) is reported but not gated. See `polaris_lfhv.limits`.

## The deployment soft wall

The lab's FR3 runs behind a soft-safety controller whose joint-velocity limits are stricter
than any of the above: hard 2.075 rad/s on j1-j4 and 2.51 on j5-j7, with a 0.5 rad/s margin
at which the controller starts pushing back, i.e. 1.575 / 2.01 rad/s. The generator's
constant ceiling is this wall (`polaris_lfhv.limits.DEPLOY_V_SOFT`), which can only make the
gate safer. The values were transcribed from the robot's `config/fr3/franka_hardware.yaml`
and should be re-read on the robot before any deployment; the 2026-08-14 test did not do
this and ran on the transcription.

## Applying FR3 limits in the simulator

Done at runtime, without editing the USD: IsaacLab 2.3.0's Articulation has
`write_joint_position_limit_to_sim(limits, joint_ids, env_ids)`, which writes
`data.joint_pos_limits`, recomputes the soft limits and pushes the DOF limits to PhysX
(`write_joint_velocity_limit_to_sim` and `write_joint_effort_limit_to_sim` exist too). The
official `noninstanceable.usd` stays untouched and the stock behaviour is one flag away
(`PourMustardEnv(fr3_limits=False)`).

## Appendix: joint order

`robot.data.joint_names` runs panda_joint1..7, then `finger_joint`, then five mimic joints
(right_outer_knuckle_joint, left/right_inner_finger_joint, left/right_inner_finger_knuckle_joint).
The first seven action dimensions map to panda_joint1-7 in that order; the eighth is the
binary `finger_joint` command (> 0.5 closes; closed angle π/4, open 0). `finger_joint`'s
position limits in the sim are 0 to 0.785398, i.e. 0 to π/4.
