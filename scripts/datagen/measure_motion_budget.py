"""What motion budget do trajectories the real arm actually executed use, at 15 Hz?

`synthesize_trajectories.py` refuses reference trajectories whose finite differences leave
FR3's motion envelope. Before trusting any of those thresholds on synthetic data they have
to be checked against trajectories that are executable by construction: the 2026-02-09
teleop episodes, whose `joint_position` column is the measured qpos of this arm at 15 Hz,
exported by `scripts/real2sim/export_lerobot_qpos.py`.

Three limits are checked, and the answers differ:

- velocity, against libfranka's position-dependent envelope
  (`computeUpper/LowerLimitsJointVelocity`, read verbatim from
  libfranka include/franka/rate_limiting.h) capped by this robot's deployment soft
  wall. Real peak 1.39 rad/s = 0.69 of the wall, and the envelope never blocks a real step,
  so the velocity gate is safe to enforce.
- acceleration, against libfranka's 10 rad/s^2. Real data BREAKS it: max 20.77 rad/s^2,
  three of eight episodes above 10, p99 1.16 and p99.9 4.60. The spec value is a
  continuous-time limit on a 1 kHz control loop; twice-differencing a 15 Hz waypoint stream
  does not measure it. Enforcing 10 here would throw away trajectories the robot has run,
  so the generator's ceiling is the measured maximum instead.
- jerk, against libfranka's 5000 rad/s^3. Real max 513, an order of magnitude under, so it
  never binds at this rate and the generator reports it without gating on it.

A fourth number falls out and matters for grasp selection: j6 stays within [1.12, 2.86]
across every real episode, nowhere near FR3's 0.4398 lower limit, while the FR3 velocity
envelope additionally forbids downward motion below 0.5521. Synthesised episodes that drive
j6 to its limit are not doing what the human did.

    $POLARIS_ROOT/.venv/bin/python scripts/datagen/measure_motion_budget.py \
        --episodes $WORK/feb_qpos
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np

# libfranka include/franka/rate_limiting.h; same constants as synthesize_trajectories.py
V_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
V_C = np.array([0.30, 0.20, 0.20, 0.30, 0.35, 0.35, 0.35])
V_K = np.array([12.0, 5.17, 7.00, 8.00, 34.0, 11.0, 34.0])
V_A_HI = np.array([2.75010, 1.79180, 2.90650, -0.14580, 2.81010, 4.52050, 3.01960])
V_A_LO = np.array([2.75010, 1.79180, 2.90650, 3.04810, 2.81010, -0.54092, 3.01960])
# deployment execution layer of this robot, see docs/provenance.md
SOFT = np.array([1.575, 1.575, 1.575, 1.575, 2.01, 2.01, 2.01])
SPEC_ACC, SPEC_JERK = 10.0, 5000.0


def velocity_window(q):
    hi = np.minimum(V_MAX, np.maximum(0.0, -V_C + np.sqrt(np.maximum(0.0, V_K * (V_A_HI - q)))))
    lo = np.maximum(-V_MAX, np.minimum(0.0, V_C - np.sqrt(np.maximum(0.0, V_K * (V_A_LO + q)))))
    return np.maximum(lo, -SOFT), np.minimum(hi, SOFT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", default=os.path.join(os.environ.get("WORK", "work"), "feb_qpos"),
                   help="directory of episode npz written by export_lerobot_qpos.py")
    p.add_argument("--rate", type=float, default=15.0)
    p.add_argument("--out", default=None, help="write the summary as json")
    args = p.parse_args()

    files = sorted(Path(args.episodes).glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no episode npz under {args.episodes}")

    rows, pooled = [], {"v": [], "a": [], "j": []}
    print(f"{'episode':>16} {'steps':>6} {'|v|max':>7} {'v/env':>7} {'|a|max':>7} "
          f"{'a/spec':>7} {'|j|max':>8} {'j/spec':>7}  j6 range")
    for f in files:
        q = np.load(f)["qpos"][:, :7]
        dq = np.diff(q, axis=0) * args.rate
        lo, hi = velocity_window(q[:-1])
        ratio = np.where(dq >= 0, dq / np.maximum(hi, 1e-9), dq / np.minimum(lo, -1e-9))
        ddq = np.diff(dq, axis=0) * args.rate
        dddq = np.diff(ddq, axis=0) * args.rate
        row = {"episode": f.stem, "steps": int(len(q)),
               "v_max": float(np.abs(dq).max()), "v_over_envelope": float(ratio.max()),
               "a_max": float(np.abs(ddq).max()), "j_max": float(np.abs(dddq).max()),
               "j6_min": float(q[:, 5].min()), "j6_max": float(q[:, 5].max())}
        rows.append(row)
        pooled["v"].append(np.abs(dq))
        pooled["a"].append(np.abs(ddq))
        pooled["j"].append(np.abs(dddq))
        print(f"{row['episode']:>16} {row['steps']:>6} {row['v_max']:7.2f} "
              f"{row['v_over_envelope']:7.2f} {row['a_max']:7.2f} "
              f"{row['a_max'] / SPEC_ACC:7.2f} {row['j_max']:8.0f} "
              f"{row['j_max'] / SPEC_JERK:7.2f}  [{row['j6_min']:.3f}, {row['j6_max']:.3f}]")

    summary = {"episodes": rows, "percentiles": {}}
    print("\npooled over every step and joint:")
    for key, name, spec in (("v", "velocity rad/s", float(SOFT.max())),
                            ("a", "accel rad/s^2", SPEC_ACC),
                            ("j", "jerk rad/s^3", SPEC_JERK)):
        x = np.concatenate(pooled[key])
        pct = np.percentile(x, [50, 95, 99, 99.9, 100])
        summary["percentiles"][name] = {"p50": pct[0], "p95": pct[1], "p99": pct[2],
                                        "p99.9": pct[3], "max": pct[4], "reference": spec}
        print(f"  {name:<16} p50 {pct[0]:8.3f}  p95 {pct[1]:8.3f}  p99 {pct[2]:8.3f}  "
              f"p99.9 {pct[3]:8.3f}  max {pct[4]:9.3f}   (reference {spec})")

    over = sum(1 for r in rows if r["a_max"] > SPEC_ACC)
    print(f"\n{over} of {len(rows)} real episodes exceed libfranka's {SPEC_ACC} rad/s^2 when "
          f"their 15 Hz positions are differenced twice; the generator's acceleration "
          f"ceiling is the measured maximum {max(r['a_max'] for r in rows):.2f} instead.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
