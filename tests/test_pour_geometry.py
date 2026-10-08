"""The pour criterion's geometry, on hand-built poses and on a synthetic object track."""
import numpy as np
import pytest

from polaris_lfhv.pour_geometry import DwellCounter, pour_condition, pour_measurements, rotate

UPRIGHT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])   # +90 deg about x: mesh +Y -> world +Z
MOUTH, RIM, RADIUS = 0.166, 0.1152, 0.0468                   # bottle height, cup rim, cup radius


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                     w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2])


def tilted(theta_deg):
    """A standing bottle tipped by theta towards +y (rotation about -x on top of UPRIGHT)."""
    t = np.radians(theta_deg)
    about_minus_x = np.array([np.cos(t / 2), -np.sin(t / 2), 0.0, 0.0])
    return quat_mul(about_minus_x, UPRIGHT)


def test_rotate_identity_and_quarter_turn():
    v = np.array([0.3, -0.2, 0.9])
    assert rotate([1, 0, 0, 0], v) == pytest.approx(v)
    assert rotate(UPRIGHT, [0, 1, 0]) == pytest.approx([0, 0, 1], abs=1e-12)
    assert rotate(UPRIGHT, [0, 0, 1]) == pytest.approx([0, -1, 0], abs=1e-12)


def test_upright_bottle_has_zero_tilt_and_mouth_at_its_top():
    tilt, d_xy, above = pour_measurements([0.4, -0.1, 0.0], UPRIGHT, [0.4, 0.15, 0.0], MOUTH, RIM)
    assert tilt == pytest.approx(0.0, abs=1e-9)
    assert d_xy == pytest.approx(0.25)
    assert above == pytest.approx(MOUTH - RIM)


def test_horizontal_bottle_reads_ninety_degrees():
    tilt, _, _ = pour_measurements([0, 0, 0], tilted(90), [0, 0, 0], MOUTH, RIM)
    assert tilt == pytest.approx(90.0)


def test_demo_numbers_satisfy_the_criterion():
    # measured on the demonstration: tilt 83-84 deg, mouth 0.019-0.022 m from the cup axis,
    # 0.063-0.070 m above the rim (docs/results.md)
    assert pour_condition(84.0, 0.020, 0.065, RADIUS)


@pytest.mark.parametrize("tilt, d_xy, above", [
    (30.0, 0.020, 0.065),    # not tilted enough
    (84.0, 0.050, 0.065),    # mouth outside the cup's radius
    (84.0, 0.020, -0.010),   # mouth below the rim: beside the cup, not over it
    (84.0, 0.020, 0.350),    # waved high overhead
])
def test_each_condition_can_fail_alone(tilt, d_xy, above):
    assert not pour_condition(tilt, d_xy, above, RADIUS)


def test_dwell_counter_needs_consecutive_steps():
    d = DwellCounter(3)
    assert [d.update(x) for x in (True, True, False, True, True, True, True)] == \
        [False, False, False, False, False, True, True]
    assert d.best == 4
    d.reset()
    assert d.held == 0 and d.best == 0


def synthetic_track(cup, n_pour=30, n_hold=20, theta_max=84.0):
    """Bottle poses for a pour over `cup`: lifted and carried, then tipped, then held."""
    poses = []
    pour_origin = np.array([cup[0] + 0.02, cup[1] - MOUTH * np.sin(np.radians(theta_max)), cup[2] + 0.16])
    for k in range(n_pour + n_hold):
        theta = theta_max * min(k, n_pour) / n_pour
        poses.append((pour_origin, tilted(theta)))
    return poses


def test_criterion_fires_after_the_dwell_on_a_synthetic_pour():
    cup = np.array([0.47, 0.155, -0.019])
    dwell = DwellCounter(15)
    first_true, fired = None, None
    for step, (pos, quat) in enumerate(synthetic_track(cup)):
        tilt, d_xy, above = pour_measurements(pos, quat, cup, MOUTH, RIM)
        now = pour_condition(tilt, d_xy, above, RADIUS)
        if now and first_true is None:
            first_true = step
        if dwell.update(now) and fired is None:
            fired = step
    assert first_true is not None and fired is not None
    assert fired == first_true + 15 - 1
    # the mouth ends up over the cup, above the rim, with the demo-like tilt
    tilt, d_xy, above = pour_measurements(*synthetic_track(cup)[-1], cup, MOUTH, RIM)
    assert tilt == pytest.approx(84.0)
    assert d_xy < RADIUS and 0 < above < 0.30
