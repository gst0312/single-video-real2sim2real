"""Write examples/data/sample_episode.json: a small synthetic episode for the CPU example.

The joint trajectory and the object track are hand-built, not an IK solution of the real
pipeline, and they are not tied to each other; they exist to exercise the kinematic gate and
the pour criterion on something small enough to ship. Real episodes come out of
`scripts/datagen/synthesize_trajectories.py` (joints, 15 Hz, ~270-970 steps) and
`scripts/datagen/replay_reference_episode.py` (object poses from physics), neither of which
runs without jax or Isaac.

    python examples/make_sample_episode.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polaris_lfhv.limits import HOME, RATE_HZ, smoothstep  # noqa: E402

UPRIGHT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])   # mesh +Y up -> world +Z
CUP = np.array([0.47, 0.155, -0.019])                          # a layout from the sampler's range
BOTTLE_START = np.array([0.45, -0.10, -0.019])
MOUTH, RIM, RADIUS = 0.166, 0.1152, 0.0468                     # bottle height, cup rim and radius (m)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                     w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2])


def tipped(theta_deg):
    """Standing bottle tipped by theta towards +y (towards the cup)."""
    t = np.radians(theta_deg)
    return quat_mul(np.array([np.cos(t / 2), -np.sin(t / 2), 0.0, 0.0]), UPRIGHT)


def segment(q_from, q_to, n):
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    return q_from[None] + smoothstep(u) * (q_to - q_from)[None]


def main():
    # joints: home -> pregrasp (30), hold while the gripper closes (10), carry (40),
    # pour by rolling the wrist (30), hold (20)
    pregrasp = HOME + np.array([0.3, 0.25, -0.1, 0.35, 0.1, 0.4, 0.8])
    carried = pregrasp + np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    poured = carried + np.array([0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 1.5])
    qpos = np.concatenate([segment(HOME, pregrasp, 30), np.tile(pregrasp, (10, 1)),
                           segment(pregrasp, carried, 40), segment(carried, poured, 30),
                           np.tile(poured, (20, 1))])
    n = len(qpos)
    close_step = 36
    gripper = (np.arange(n) >= close_step).astype(float)

    # object track: the bottle sits until it is grasped (step 40), is lifted and carried over
    # the cup during the carry (40..79), tipped to 84 degrees during the pour (80..109), held
    pour_origin = np.array([CUP[0] + 0.02, CUP[1] - MOUTH * np.sin(np.radians(84.0)), CUP[2] + 0.16])
    bottle = []
    for k in range(n):
        if k < 40:
            pos, quat = BOTTLE_START, UPRIGHT
        elif k < 80:
            u = smoothstep((k - 40) / 40)
            pos, quat = BOTTLE_START + u * (pour_origin - BOTTLE_START), UPRIGHT
        else:
            theta = 84.0 * min(k - 80, 30) / 30
            pos, quat = pour_origin, tipped(theta)
        bottle.append([*np.round(pos, 5), *np.round(quat, 6)])

    out = ROOT / "examples" / "data" / "sample_episode.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "synthetic illustration for examples/gate_and_rubric_demo.py; not an IK solution",
        "rate_hz": RATE_HZ,
        "instruction": "pour the mustard into the blue cup",
        "qpos": np.round(qpos, 5).tolist(),
        "gripper": gripper.tolist(),
        "close_step": close_step,
        "bottle_pose_wxyz": bottle,
        "cup_pose": [*CUP, *UPRIGHT],
        "geometry": {"bottle_mouth_m": MOUTH, "cup_rim_m": RIM, "cup_radius_m": RADIUS},
    }, indent=0))
    print(f"wrote {out} ({n} steps at {RATE_HZ:g} Hz)")


if __name__ == "__main__":
    main()
