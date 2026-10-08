"""The kinematic gate: FR3-and-Panda joint limits, the libfranka velocity envelope, the
motion budget and the time-rescaling rule the synthesiser applies instead of dropping a
too-fast episode. Pure numpy; numbers are the ones documented in docs/fr3_limits.md."""
import numpy as np
import pytest

from r2s2r import limits as L


def smooth_path(start, delta, n, rate=L.RATE_HZ):
    """(n, 7) smoothstep path from start to start + delta."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    return start[None] + L.smoothstep(u) * np.asarray(delta)[None]


def test_intersection_is_the_tighter_of_fr3_and_panda():
    # FR3 is tighter at j4 upper, j5 both sides and j6 lower; Panda everywhere else
    assert L.LIMITS_HIGH[3] == pytest.approx(-0.1169)   # FR3 j4 upper (Panda: -0.0698)
    assert L.LIMITS_LOW[4] == pytest.approx(-2.8763)    # FR3 j5 (Panda: -2.8973)
    assert L.LIMITS_LOW[5] == pytest.approx(0.4398)     # FR3 j6 lower (Panda: -0.0175)
    assert L.LIMITS_LOW[1] == pytest.approx(-1.7628)    # Panda j2 (FR3: -1.8361)
    assert L.LIMITS_HIGH[6] == pytest.approx(2.8973)    # Panda j7 (FR3: 3.0508)
    assert np.all(L.LIMITS_LOW < L.LIMITS_HIGH)


def test_home_pose_is_inside_the_intersection():
    assert L.within_limits(L.HOME[None])
    assert L.within_limits(np.tile(L.HOME, (5, 2, 1))).tolist() == [True, True]


def test_within_limits_flags_a_single_violation():
    q = np.tile(L.HOME, (10, 1))
    assert L.within_limits(q)
    q[4, 5] = 0.40          # j6 below FR3's 0.4398 (Panda alone would allow it)
    assert not L.within_limits(q)


def test_velocity_window_is_the_deployment_ceiling_far_from_the_limits():
    lo, hi = L.fr3_velocity_window(L.HOME)
    assert hi == pytest.approx(L.VEL_CEILING)
    assert lo == pytest.approx(-L.VEL_CEILING)
    # the ceiling is the soft-safety wall, which is below FR3's catalogue numbers everywhere
    assert np.all(L.VEL_CEILING < L.FR3_V_MAX)
    assert L.VEL_CEILING.tolist() == pytest.approx([1.575] * 4 + [2.01] * 3)


def test_velocity_window_closes_at_the_envelope_and_shrinks_towards_it():
    _, hi_at_envelope = L.fr3_velocity_window(L.ENVELOPE_HIGH)
    lo_at_envelope, _ = L.fr3_velocity_window(L.ENVELOPE_LOW)
    assert hi_at_envelope == pytest.approx(np.zeros(7), abs=1e-9)
    assert lo_at_envelope == pytest.approx(np.zeros(7), abs=1e-9)
    # past the envelope the allowed speed stays zero rather than going negative
    _, hi_beyond = L.fr3_velocity_window(L.ENVELOPE_HIGH + 0.1)
    assert hi_beyond == pytest.approx(np.zeros(7), abs=1e-9)
    # monotone: closer to the limit, less speed allowed
    q_mid = 0.5 * (L.HOME + L.ENVELOPE_HIGH)
    q_near = 0.9 * L.ENVELOPE_HIGH + 0.1 * L.HOME
    assert np.all(L.fr3_velocity_window(q_near)[1] <= L.fr3_velocity_window(q_mid)[1] + 1e-12)


def test_j6_downward_motion_is_blocked_before_its_position_limit():
    # documented in docs/fr3_limits.md: j6's FR3 limit is 0.4398 but libfranka's envelope
    # already forbids downward motion below 0.5521
    assert L.ENVELOPE_LOW[5] == pytest.approx(0.5521, abs=1e-3)
    assert L.SAFE_LOW[5] > L.LIMITS_LOW[5]


def test_static_trajectory_uses_no_budget():
    q = np.tile(L.HOME, (30, 3, 1))
    vel, acc, jerk = L.motion_ratios(q)
    assert vel.shape == (3,)
    assert np.all(vel == 0) and np.all(acc == 0) and np.all(jerk == 0)


def test_gate_rejects_a_too_fast_episode_and_the_rescale_fixes_it():
    delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])     # j7 roll, as in the pour
    fast = smooth_path(L.HOME, delta, n=12)                 # 1.5*1.5/12*15 = 2.8 rad/s
    g = L.kinematic_gate(fast)
    assert g["limits"] is True
    assert g["vel"] is False
    assert g["vel_ratio"] > 1.0
    assert g["time_scale_needed"] == pytest.approx(max(g["vel_ratio"], np.sqrt(g["acc_ratio"])))
    # the synthesiser's remedy: the same motion sampled more finely by that factor
    slow = smooth_path(L.HOME, delta, n=int(np.ceil(12 * g["time_scale_needed"] * 1.05)))
    g2 = L.kinematic_gate(slow)
    assert g2["vel"] and g2["acc"] and g2["limits"]
    assert g2["time_scale_needed"] == 1.0


def test_gate_batch_shapes_and_per_episode_verdicts():
    ok = smooth_path(L.HOME, [0.3] * 7, n=40)
    bad = smooth_path(L.HOME, [0.5] * 7, n=5)   # 0.17 rad per step = 2.6 rad/s
    q = np.stack([ok, bad[np.linspace(0, 4, 40).round().astype(int)]], axis=1)  # (40, 2, 7)
    g = L.kinematic_gate(q)
    assert g["vel"].shape == (2,)
    assert g["vel"][0] and not g["vel"][1]
    assert g["jerk_ratio"].shape == (2,)


def test_home_ramp_starts_at_home_and_stays_inside_its_budget():
    q_first = (L.HOME + np.array([0.4, -0.3, 0.2, 0.5, -0.2, 0.6, 1.0]))[None]   # (1, 7)
    ramp = L.home_ramp(q_first)
    assert ramp.shape[1:] == (1, 7)
    assert ramp[0, 0] == pytest.approx(L.HOME)
    # last ramp sample is one step before q_first, so appending q_first closes the path
    path = np.concatenate([ramp, q_first[None]], axis=0)
    vel, acc, _ = L.motion_ratios(path)
    assert vel[0] <= L.RAMP_BUDGET + 0.02
    assert acc[0] <= L.RAMP_BUDGET + 0.05
    assert L.within_limits(path)[0]
