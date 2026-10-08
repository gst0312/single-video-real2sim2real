"""FR3 motion limits and the kinematic gate, in plain numpy.

Everything here is importable without jax, Isaac or a GPU, which is what lets it be unit
tested and run in the CPU example. `scripts/datagen/synthesize_trajectories.py` imports
these names instead of defining them; `scripts/datagen/measure_motion_budget.py` and
`scripts/eval/log_action_rates.py` carry the same constants for the same reasons.

Sources (details in docs/fr3_limits.md):

- Joint position limits are the FR3-and-Panda intersection: FR3 from franka_description
  2.8.1 `robots/fr3/joint_limits.yaml`, Panda from the limits PolaRiS's nvidia_droid USD
  carries. Data has to be executable on the FR3 and replayable in the Panda-limited sim.
- The position-dependent velocity envelope is libfranka's
  `computeUpper/LowerLimitsJointVelocity` (`include/franka/rate_limiting.h`):
  min(v_max, max(0, -c + sqrt(max(0, k*(a_hi - q))))) and its mirror. The tolerance term is
  kLimitEps = 1e-3 because kTolNumberPacketsLost is 0 for the FR3, so it is dropped.
- The constant ceiling is the deployment execution layer's soft-safety wall (1.575 rad/s
  J1-4, 2.01 J5-7), not FR3's catalogue 2.62/5.26/4.18: it is stricter than every official
  number, so it can only make the gate safer. It was transcribed from the robot's
  `config/fr3/franka_hardware.yaml` (hard 2.075 / 2.51 minus the controller's 0.5 margin)
  and should be re-read on the robot before any deployment.
- The acceleration ceiling is measured, not taken from the spec. libfranka's 10 rad/s^2
  applies to its 1 kHz loop; a 15 Hz waypoint stream differenced twice is a different
  quantity, and `measure_motion_budget.py` shows real teleoperated episodes reach
  20.77 rad/s^2 (three of eight above 10). Gating at 10 would reject trajectories the robot
  has demonstrably executed, so the ceiling is that measured maximum. The same measurement
  validates the velocity gate (real peak 1.39 rad/s = 0.69 of the soft wall) and retires
  the jerk gate (real max 513 against libfranka's 5000), which is reported but not enforced.
"""

import numpy as np

RATE_HZ = 15.0
#: PolaRiS / DROID reset pose, radians: (0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0)
HOME = np.array([0.0, -0.6283, 0.0, -2.5133, 0.0, 1.885, 0.0])

# Joint position limits: the FR3-and-Panda intersection (docs/fr3_limits.md)
LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8763, 0.4398, -2.8973])
LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.1169, 2.8763, 3.7525, 2.8973])

# libfranka include/franka/rate_limiting.h (frankarobotics/libfranka main)
FR3_V_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_V_C = np.array([0.30, 0.20, 0.20, 0.30, 0.35, 0.35, 0.35])
FR3_V_K = np.array([12.0, 5.17, 7.00, 8.00, 34.0, 11.0, 34.0])
FR3_V_A_HI = np.array([2.75010, 1.79180, 2.90650, -0.14580, 2.81010, 4.52050, 3.01960])
FR3_V_A_LO = np.array([2.75010, 1.79180, 2.90650, 3.04810, 2.81010, -0.54092, 3.01960])
FR3_ACC_MAX = 10.0                          # kMaxJointAcceleration, at libfranka's 1 kHz
FR3_JERK_MAX = 5000.0                       # kMaxJointJerk, likewise

#: Acceleration ceiling for a 15 Hz reference stream: the maximum measured on real
#: executed trajectories (see the module docstring), not the 1 kHz spec value.
ACC_MAX = 20.8

#: Deployment execution layer of the lab's robot: soft-safety wall per joint, rad/s.
DEPLOY_V_SOFT = np.array([1.575, 1.575, 1.575, 1.575, 2.01, 2.01, 2.01])
VEL_CEILING = np.minimum(FR3_V_MAX, DEPLOY_V_SOFT)

RAMP_BUDGET = 0.5      # fraction of the motion budget the home ramp is allowed to use

# Positions where FR3's velocity envelope reaches zero, i.e. where the arm may not move
# further in that direction at any speed: solving `-c + sqrt(k*(a_hi - q)) = 0` and its
# mirror. Slowing down cannot rescue a trajectory that needs to cross these, so grasp
# screening keeps clear of them. For j6 this matters most: its FR3 position limit is
# 0.4398 but downward motion is already blocked at 0.5521.
ENVELOPE_HIGH = FR3_V_A_HI - FR3_V_C ** 2 / FR3_V_K
ENVELOPE_LOW = FR3_V_C ** 2 / FR3_V_K - FR3_V_A_LO
SAFE_LOW = np.maximum(LIMITS_LOW, ENVELOPE_LOW)
SAFE_HIGH = np.minimum(LIMITS_HIGH, ENVELOPE_HIGH)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def fr3_velocity_window(q):
    """(lower, upper) joint velocity allowed at joint position q, in rad/s.

    libfranka's position-dependent envelope, then clipped to VEL_CEILING. Vectorised over
    any leading shape; q's last axis is the seven arm joints.
    """
    q = np.asarray(q, dtype=float)
    hi = np.minimum(FR3_V_MAX, np.maximum(
        0.0, -FR3_V_C + np.sqrt(np.maximum(0.0, FR3_V_K * (FR3_V_A_HI - q)))))
    lo = np.maximum(-FR3_V_MAX, np.minimum(
        0.0, FR3_V_C - np.sqrt(np.maximum(0.0, FR3_V_K * (FR3_V_A_LO + q)))))
    return np.maximum(lo, -VEL_CEILING), np.minimum(hi, VEL_CEILING)


def motion_ratios(qpos, rate=RATE_HZ):
    """Per-episode overshoot of the velocity, acceleration and jerk budgets.

    qpos is (T, b, 7). Returns three (b,) arrays; a value <= 1 means that budget is met.
    The velocity window is evaluated at both ends of every step and the tighter one is
    used, because the envelope shrinks towards the joint limits.
    """
    qpos = np.asarray(qpos, dtype=float)
    dq = np.diff(qpos, axis=0) * rate
    lo_a, hi_a = fr3_velocity_window(qpos[:-1])
    lo_b, hi_b = fr3_velocity_window(qpos[1:])
    hi = np.maximum(np.minimum(hi_a, hi_b), 1e-9)
    lo = np.minimum(np.maximum(lo_a, lo_b), -1e-9)
    vel = np.where(dq >= 0, dq / hi, dq / lo)

    ddq = np.diff(dq, axis=0) * rate
    dddq = np.diff(ddq, axis=0) * rate
    return (vel.max(axis=(0, 2)),
            np.abs(ddq).max(axis=(0, 2)) / ACC_MAX,
            np.abs(dddq).max(axis=(0, 2)) / FR3_JERK_MAX)


def home_ramp(q_first, rate=RATE_HZ, budget=RAMP_BUDGET):
    """Smoothstep transit from HOME to the first synthesised pose, inside the budget.

    r2r2r lets its online PD controller chase the first target; replaying reference actions
    open loop has no controller to lean on, so the transit is explicit. A linear ramp is
    not usable: it starts and ends with a velocity step, i.e. an unbounded acceleration.
    For a smoothstep over n steps the peaks are |dq| = 1.5*delta/n*rate and
    |ddq| = 6*delta/n^2*rate^2, so n follows from both budgets. q_first is (b, 7); returns
    (n_ramp, b, 7) with HOME first and the step before q_first last.
    """
    q_first = np.asarray(q_first, dtype=float)
    delta = np.abs(q_first - HOME)                       # (b, 7)
    n_vel = 1.5 * delta * rate / (budget * VEL_CEILING)
    n_acc = rate * np.sqrt(6.0 * delta / (budget * ACC_MAX))
    n = int(np.ceil(max(n_vel.max(), n_acc.max()))) + 1
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None, None]
    return HOME[None, None] + smoothstep(u) * (q_first[None] - HOME[None, None])


def within_limits(qpos):
    """Per-episode flag: every joint of every step inside the FR3-and-Panda intersection.

    qpos is (T, b, 7) or (T, 7); the result is (b,) or a scalar bool.
    """
    qpos = np.asarray(qpos, dtype=float)
    ok = (qpos >= LIMITS_LOW) & (qpos <= LIMITS_HIGH)
    return ok.all(axis=(0, -1)) if qpos.ndim == 3 else bool(ok.all())


def kinematic_gate(qpos, rate=RATE_HZ):
    """The motion-budget part of the synthesiser's gate on a joint trajectory.

    qpos is (T, 7) for one episode or (T, b, 7) for a batch. Returns a dict with per-episode
    booleans `limits`, `vel`, `acc` and the ratios `vel_ratio`, `acc_ratio`, `jerk_ratio`
    (jerk is reported, not gated), plus `time_scale_needed`: the factor the episode would
    have to be slowed down by to pass the velocity and acceleration budgets (1.0 when it
    already passes). The IK-error and cup-clearance checks need the solver and the task
    geometry and stay in the synthesiser.
    """
    qpos = np.asarray(qpos, dtype=float)
    single = qpos.ndim == 2
    if single:
        qpos = qpos[:, None, :]
    vel, acc, jerk = motion_ratios(qpos, rate)
    need = np.maximum(1.0, np.maximum(vel, np.sqrt(np.maximum(acc, 0.0))))
    out = {
        "limits": within_limits(qpos),
        "vel": vel <= 1.0,
        "acc": acc <= 1.0,
        "vel_ratio": vel,
        "acc_ratio": acc,
        "jerk_ratio": jerk,
        "time_scale_needed": need,
    }
    if single:
        out = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in out.items()}
        out = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in out.items()}
    return out
