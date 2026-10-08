"""What does the generated dataset actually span?

Pass rates say the episodes are executable. They say nothing about whether the set is
worth training on: a hundred episodes that all start in the same spot, held by the same
grasp, sweeping the same arc, teach a policy one trajectory. So this measures the spread -
layout, grasp, joint space, end-effector workspace - and prints it next to the thing each
one should be compared against (the demonstration, the sampler's range, the joint limits).

Reads the synthesised episodes (`qpos`, `grasp_id`, `condition_id`, `time_scale`) and, if
the physics reports are given, restricts to the episodes that passed the gate - which is
what the dataset actually contains.

    $TRAJ_VENV/bin/python scripts/datagen/analyse_dataset.py \
        --episodes $WORK/traj/prod \
                   $WORK/traj/prod2 \
        --reports  $WORK/traj/data \
                   $WORK/traj/data2 \
        --conditions $POLARIS_ROOT/PolaRiS-Hub/pour_mustard/initial_conditions.json \
        --plot $WORK/dataset_coverage.png
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf

# the pure-numpy pieces live in the package so they can be tested without the heavy
# dependencies; make the repository's src importable when the package is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polaris_lfhv.limits import LIMITS_HIGH, LIMITS_LOW  # noqa: E402
from polaris_lfhv.physics_gate import CUP_MAX, GAP_MAX, LIFT_MIN, passes_physics_gate  # noqa: E402


def passed(report_dir, stem, lift_min=LIFT_MIN, gap_max=GAP_MAX, cup_max=CUP_MAX):
    """The physics gate for one episode, from the report beside it.

    The report directory is paired with its own episode directory rather than searched:
    the two synthesis batches use different seeds but the same episode names, so a search
    would judge a second-batch episode by a first-batch report.
    """
    f = Path(report_dir) / f"{stem}.json"
    if not f.exists():
        return None
    return passes_physics_gate(json.loads(f.read_text()), lift_min, gap_max, cup_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", required=True,
                   help="episode directories; pair each with the same-index --reports entry")
    p.add_argument("--reports", nargs="*", default=[])
    p.add_argument("--conditions", required=True)
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    conds = json.load(open(args.conditions))["poses"]
    kin = JaxKinTree.from_urdf(load_urdf("fr3_description"))
    idx = kin.joint_names.index("fr3_joint8")

    rows, skipped = [], 0
    for i, d in enumerate(args.episodes):
        rep = args.reports[i] if i < len(args.reports) else None
        for f in sorted(Path(d).glob("*.npz")):
            e = np.load(f)
            if "qpos" not in e:
                continue
            if rep and passed(rep, f.stem) is not True:
                skipped += 1
                continue
            q = e["qpos"].astype(np.float64)
            qj = jnp.concatenate([jnp.asarray(q),
                                  jnp.zeros((len(q), kin.num_actuated_joints - 7))], axis=1)
            ee = np.asarray(jaxlie.SE3(kin.forward_kinematics(qj)[:, idx]).translation())
            rows.append({"cond": int(e["condition_id"]), "grasp": int(e["grasp_id"]),
                         "scale": float(e["time_scale"]), "steps": len(q),
                         "q": q, "ee": ee})
    if not rows:
        raise SystemExit("no episodes matched")
    print(f"{len(rows)} episodes in the dataset ({skipped} dropped by the physics gate)\n")

    # --- layout coverage -------------------------------------------------------------
    used = Counter(r["cond"] for r in rows)
    print(f"initial conditions: {len(used)} of {len(conds)} represented")
    counts = np.array(sorted(used.values()))
    print(f"  episodes per condition: min {counts.min()}, median {int(np.median(counts))}, "
          f"max {counts.max()}")
    missing = sorted(set(range(len(conds))) - set(used))
    print(f"  conditions with no data: {len(missing)}" + (f" -> {missing}" if missing else ""))

    bxy = np.array([conds[r["cond"]]["mustard"][:2] for r in rows])
    cxy = np.array([conds[r["cond"]]["blue_cup"][:2] for r in rows])
    print(f"\nbottle start spread : x {bxy[:, 0].min():.3f}-{bxy[:, 0].max():.3f} "
          f"y {bxy[:, 1].min():.3f}-{bxy[:, 1].max():.3f} m "
          f"(sampler drew +-0.05 about the demo)")
    print(f"cup spread          : x {cxy[:, 0].min():.3f}-{cxy[:, 0].max():.3f} "
          f"y {cxy[:, 1].min():.3f}-{cxy[:, 1].max():.3f} m (+-0.03)")

    # --- grasp diversity -------------------------------------------------------------
    g = Counter(r["grasp"] for r in rows)
    top = g.most_common(1)[0]
    print(f"\ngrasps: {len(g)} distinct of 2880 candidates; most used appears "
          f"{top[1]} times ({100 * top[1] / len(rows):.0f}% of episodes)")
    print(f"  episodes per grasp: median {int(np.median(sorted(g.values())))}, "
          f"max {max(g.values())}")

    # --- joint space -----------------------------------------------------------------
    allq = np.concatenate([r["q"] for r in rows])
    print(f"\njoint coverage, fraction of the FR3-and-Panda range each joint visits:")
    span = (allq.max(0) - allq.min(0)) / (LIMITS_HIGH - LIMITS_LOW)
    for i in range(7):
        print(f"  j{i+1}: {np.degrees(allq[:, i].min()):7.1f} to "
              f"{np.degrees(allq[:, i].max()):7.1f} deg   {span[i] * 100:4.1f}% of its limit range")

    # --- end effector workspace ------------------------------------------------------
    allee = np.concatenate([r["ee"] for r in rows])
    print(f"\nend-effector path box: x {allee[:, 0].min():.3f}-{allee[:, 0].max():.3f}, "
          f"y {allee[:, 1].min():.3f}-{allee[:, 1].max():.3f}, "
          f"z {allee[:, 2].min():.3f}-{allee[:, 2].max():.3f} m")
    # how much of that box the paths actually visit, on a 2 cm grid in xy
    cell = 0.02
    occ = {(int(x // cell), int(y // cell)) for x, y in allee[:, :2]}
    area = len(occ) * cell * cell
    box = (allee[:, 0].max() - allee[:, 0].min()) * (allee[:, 1].max() - allee[:, 1].min())
    print(f"  xy footprint actually visited: {area * 1e4:.0f} cm2 over {len(occ)} cells "
          f"({100 * area / box:.0f}% of the bounding box)")

    print(f"\nepisode length: {min(r['steps'] for r in rows)} to "
          f"{max(r['steps'] for r in rows)} steps, median "
          f"{int(np.median([r['steps'] for r in rows]))}")
    sc = Counter(round(r["scale"], 2) for r in rows)
    print(f"time scales in use: " + ", ".join(f"{k}x:{v}" for k, v in sorted(sc.items())))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
        for r in rows:
            ax[0].plot(r["ee"][:, 0], r["ee"][:, 1], lw=0.4, alpha=0.25, color="#2C6A8C")
        ax[0].scatter(bxy[:, 0], bxy[:, 1], s=14, color="#B07E10", label="bottle start", zorder=3)
        ax[0].scatter(cxy[:, 0], cxy[:, 1], s=14, color="#2C7A57", label="cup", zorder=3)
        ax[0].set_title(f"end-effector paths, {len(rows)} episodes")
        ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
        ax[0].legend(loc="best", fontsize=8); ax[0].set_aspect("equal"); ax[0].grid(alpha=0.2)

        ax[1].bar(range(len(conds)), [used.get(i, 0) for i in range(len(conds))],
                  color="#2C6A8C")
        ax[1].set_title("episodes per initial condition")
        ax[1].set_xlabel("condition"); ax[1].set_ylabel("episodes"); ax[1].grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
