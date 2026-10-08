"""Rank synthesised episodes by how closely their motion matches the reference round.

The gates say whether an episode is executable. They say nothing about whether it moves
the way the demonstration is poured from, and that is the thing that went wrong twice: a
top grasp passes every limit while pouring by swinging the arm through the cup.

So this scores the shape of the motion, in the same four numbers measured on an earlier
lab pipeline's generated episodes for this task (82 episodes, FR3 forward kinematics on
their `joint_position`):

    approach below horizontal at the grasp   23 - 29 deg
    end-effector height at the grasp         0.126 - 0.142 m
    palm-down at the final frame             0.98 - 1.00
    last quarter carried by the wrist        j5 65-74, j7 66-67 deg, while j1 2.7-6.5,
                                             j2 1-12 and j4 1-12 barely move

Each episode gets a penalty per number: zero inside the reference band, and outside it the
distance from the band divided by a tolerance, so the numbers are comparable. Lowest total
wins. Nothing here is a gate - it is a ranking, used to pick which episode to look at and
to see whether a parameter change moved the distribution the right way.

    $TRAJ_VENV/bin/python scripts/datagen/score_against_reference.py \
        --episodes $WORK/traj/tight
"""

import argparse
import glob

import jax.numpy as jnp
import jaxlie
import numpy as np
from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf

# measured on the earlier pipeline's generated qpos; (low, high, tolerance) in the metric's own unit
REFERENCE = {
    "approach_deg": (23.0, 29.0, 8.0),
    "grasp_height_m": (0.126, 0.142, 0.03),
    "palm_down_end": (0.98, 1.00, 0.15),
    "wrist_j5_deg": (65.0, 74.0, 15.0),
    "wrist_j7_deg": (66.0, 67.0, 20.0),
    "base_j1_deg": (0.0, 6.5, 8.0),
    "arm_j2_deg": (0.0, 12.0, 10.0),
    "arm_j4_deg": (0.0, 12.0, 10.0),
}


def measure(q7, kin, idx):
    q = jnp.concatenate([jnp.asarray(q7), jnp.zeros((len(q7), kin.num_actuated_joints - 7))],
                        axis=1)
    T = jaxlie.SE3(kin.forward_kinematics(q)[:, idx])
    pos = np.asarray(T.translation())
    approach = np.asarray(T.rotation().as_matrix())[:, :, 2]
    down = -approach[:, 2]
    lo = int(np.argmin(pos[: len(q7) // 2, 2]))       # the grasp: lowest point early on
    tail = slice(int(len(q7) * 0.75), len(q7))
    travel = np.degrees(np.abs(q7[tail][-1] - q7[tail][0]))
    return {
        "approach_deg": float(np.degrees(np.arcsin(np.clip(down[lo], -1, 1)))),
        "grasp_height_m": float(pos[lo, 2]),
        "palm_down_end": float(down[-1]),
        "wrist_j5_deg": float(travel[4]), "wrist_j7_deg": float(travel[6]),
        "base_j1_deg": float(travel[0]),
        "arm_j2_deg": float(travel[1]), "arm_j4_deg": float(travel[3]),
    }


def penalty(m):
    total = 0.0
    for k, (lo, hi, tol) in REFERENCE.items():
        v = m[k]
        total += max(0.0, lo - v, v - hi) / tol
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--top", type=int, default=6)
    args = p.parse_args()

    kin = JaxKinTree.from_urdf(load_urdf("fr3_description"))
    idx = kin.joint_names.index("fr3_joint8")

    rows = []
    for f in sorted(glob.glob(f"{args.episodes}/*.npz")):
        if f.endswith(("_obs.npz", "_bottle_track.npz")):
            continue
        d = np.load(f)
        if "qpos" not in d:
            continue
        m = measure(d["qpos"].astype(np.float64), kin, idx)
        rows.append((penalty(m), f.split("/")[-1][:-4], m, int(len(d["qpos"]))))
    rows.sort()

    print(f"{len(rows)} episodes in {args.episodes}, ranked by distance from the reference\n")
    print(f"{'episode':<18}{'score':>7}{'appr':>7}{'height':>8}{'palm':>7}"
          f"{'j5':>7}{'j7':>7}{'j1':>7}{'j2':>7}{'j4':>7}{'steps':>7}")
    for s, name, m, n in rows:
        print(f"{name:<18}{s:7.2f}{m['approach_deg']:7.0f}{m['grasp_height_m']:8.3f}"
              f"{m['palm_down_end']:7.2f}{m['wrist_j5_deg']:7.0f}{m['wrist_j7_deg']:7.0f}"
              f"{m['base_j1_deg']:7.0f}{m['arm_j2_deg']:7.0f}{m['arm_j4_deg']:7.0f}{n:7d}")
    print(f"\nreference band: " + "  ".join(
        f"{k.split('_')[0]} {lo:g}-{hi:g}" for k, (lo, hi, _) in REFERENCE.items()))
    print(f"\nbest {args.top}: " + " ".join(n for _, n, _, _ in rows[:args.top]))


if __name__ == "__main__":
    main()
