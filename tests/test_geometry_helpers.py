"""Small geometry helpers used by the asset pipeline: the Umeyama fit, the quaternion-to-
matrix conversion and area-weighted mesh sampling, and the initial-condition quaternions."""
import numpy as np
import pytest

import make_initial_conditions as mic
from polaris_lfhv.alignment import umeyama
from polaris_lfhv.pour_geometry import rotate
from polaris_lfhv.robot_links import quat_to_mat, sample_mesh


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    rot = quat_to_mat(random_quat(rng))
    scale, trans = 1.7, np.array([0.3, -0.2, 0.8])
    dst = scale * src @ rot.T + trans
    s, r, t = umeyama(src, dst)
    assert s == pytest.approx(scale)
    assert r == pytest.approx(rot, abs=1e-9)
    assert t == pytest.approx(trans, abs=1e-9)


def test_umeyama_rigid_holds_the_scale_at_one():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 3))
    rot = quat_to_mat(random_quat(rng))
    dst = src @ rot.T + 0.5
    s, r, t = umeyama(src, dst, with_scale=False)
    assert s == 1.0
    assert r == pytest.approx(rot, abs=1e-9)
    assert t == pytest.approx(np.full(3, 0.5), abs=1e-9)


def test_quat_to_mat_agrees_with_rotate_and_is_proper():
    rng = np.random.default_rng(2)
    for _ in range(5):
        q = random_quat(rng)
        v = rng.normal(size=3)
        m = quat_to_mat(q)
        assert m @ v == pytest.approx(rotate(q, v))
        assert np.linalg.det(m) == pytest.approx(1.0)
        assert m @ m.T == pytest.approx(np.eye(3), abs=1e-12)


def test_sample_mesh_points_lie_on_the_triangles():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)   # unit square
    counts, indices = np.array([4]), np.array([0, 1, 2, 3])                       # one quad, fanned
    rng = np.random.default_rng(3)
    s = sample_mesh(pts, counts, indices, 500, rng)
    assert s.shape == (500, 3)
    assert np.all(s[:, 2] == 0) and np.all(s[:, :2] >= 0) and np.all(s[:, :2] <= 1)
    # area weighting: both fan triangles are equal, so both halves of the square fill
    assert 0.35 < np.mean(s[:, 0] > s[:, 1]) < 0.65


def test_sample_mesh_handles_degenerate_input():
    rng = np.random.default_rng(4)
    flat = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)                # zero area
    assert sample_mesh(flat, np.array([3]), np.array([0, 1, 2]), 10, rng).shape == (0, 3)
    assert sample_mesh(flat, np.array([]), np.array([]), 10, rng).shape == (0, 3)


def test_initial_condition_quaternions_stand_the_mesh_upright_and_yaw_the_label():
    up = mic.rotate(mic.UPRIGHT, np.array([0.0, 1.0, 0.0]))
    assert up == pytest.approx([0, 0, 1], abs=1e-12)              # mesh +Y -> world +Z
    label = mic.rotate(mic.UPRIGHT, np.array([0.0, 0.0, 1.0]))
    assert label == pytest.approx([0, -1, 0], abs=1e-12)          # mesh +Z (label) -> world -Y
    q = mic.quat_mul(mic.yaw_quat(np.pi / 2), mic.UPRIGHT)        # +90 deg yaw on top
    assert mic.rotate(q, np.array([0.0, 1.0, 0.0])) == pytest.approx([0, 0, 1], abs=1e-12)
    assert mic.rotate(q, np.array([0.0, 0.0, 1.0])) == pytest.approx([1, 0, 0], abs=1e-12)
    assert np.linalg.norm(q) == pytest.approx(1.0)
