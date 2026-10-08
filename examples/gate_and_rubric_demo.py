"""CPU-only walk through the two gates and the task criterion, on a bundled sample.

Three things the pipeline does to every episode, run here on `data/sample_episode.json`
without jax, Isaac or a GPU:

1. the kinematic gate (`r2s2r.limits`): joint positions inside the FR3-and-Panda
   intersection, joint velocity inside libfranka's position-dependent envelope capped by the
   deployment soft wall, acceleration under the measured ceiling. A copy of the episode
   played at double speed is shown failing, and the synthesiser's remedy - slow the same
   motion down by the measured overshoot - is shown fixing it;
2. the pour criterion (`r2s2r.pour_geometry`): tilt, mouth-to-cup distance and
   mouth-above-rim per step, with the 15-step dwell, on the sample's object track;
3. the physics gate (`r2s2r.physics_gate`) on the three real replay reports under
   `results/phase3_rollouts/`, plus one synthetic failure.

    python examples/gate_and_rubric_demo.py

Runs in well under a second. The sample is synthetic (see make_sample_episode.py); the
numbers it prints are about the gates, not about the real dataset.
"""
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r2s2r import limits as L  # noqa: E402
from r2s2r.physics_gate import passes_physics_gate, physics_gate_reasons  # noqa: E402
from r2s2r.pour_geometry import DwellCounter, pour_condition, pour_measurements  # noqa: E402


def resample(qpos, factor):
    """The same path sampled `factor` times more finely (factor < 1: played faster)."""
    t_old = np.arange(len(qpos))
    t_new = np.linspace(0, len(qpos) - 1, max(2, int(round(len(qpos) * factor))))
    return np.stack([np.interp(t_new, t_old, qpos[:, j]) for j in range(7)], axis=1)


def ratio(r):
    # inside the envelope's zero-speed band the allowed velocity is ~0 and the ratio is
    # astronomically large; the synthesiser reports it the same way (it means "cannot be
    # rescued by slowing down", not "a bit too fast")
    return f"{r:.2f}x" if r < 1e3 else "blocked"


def show_gate(name, qpos, rate):
    g = L.kinematic_gate(qpos, rate)
    verdict = "PASS" if g["limits"] and g["vel"] and g["acc"] else "FAIL"
    print(f"  {name:<28} {len(qpos):4d} steps  limits {str(g['limits']):<5} "
          f"vel {ratio(g['vel_ratio'])} {'ok ' if g['vel'] else 'OVER'}  "
          f"acc {ratio(g['acc_ratio'])} {'ok ' if g['acc'] else 'OVER'}  "
          f"jerk {g['jerk_ratio']:.3f}x (reported)  -> {verdict}")
    return g


def main():
    ep = json.loads((ROOT / "examples" / "data" / "sample_episode.json").read_text())
    qpos = np.asarray(ep["qpos"])
    rate = float(ep["rate_hz"])
    print(f"sample episode: {len(qpos)} steps at {rate:g} Hz, gripper closes at step {ep['close_step']}, "
          f"instruction {ep['instruction']!r}\n")

    print("1. kinematic gate (FR3-and-Panda limits, velocity envelope, motion budget)")
    print(f"   velocity ceiling rad/s: {L.VEL_CEILING}\n   acceleration ceiling rad/s^2: {L.ACC_MAX}")
    show_gate("as synthesised", qpos, rate)
    fast = resample(qpos, 0.5)
    g = show_gate("played at 2x speed", fast, rate)
    scale = float(np.ceil(g["time_scale_needed"] * 1.05 * 4) / 4)     # the synthesiser's rounding
    show_gate(f"2x copy slowed by {scale:g}x", resample(fast, scale), rate)
    bad = qpos.copy()
    bad[60:, 5] = 0.40                                                    # j6 below FR3's 0.4398
    show_gate("j6 driven to 0.40 rad", bad, rate)
    lo, hi = L.fr3_velocity_window(qpos[-1])
    print(f"   velocity window at the final pose (rad/s): {np.round(lo, 2)} .. {np.round(hi, 2)}\n")

    print("2. pour criterion on the sample's object track (tilt > 60 deg, mouth inside the cup's")
    print("   radius, mouth above the rim by 0 .. 0.30 m, held for 15 consecutive steps)")
    geom = ep["geometry"]
    cup = np.asarray(ep["cup_pose"][:3])
    dwell = DwellCounter(15)
    first, fired = None, None
    for step, pose in enumerate(ep["bottle_pose_wxyz"]):
        tilt, d_xy, above = pour_measurements(pose[:3], pose[3:], cup, geom["bottle_mouth_m"], geom["cup_rim_m"])
        now = pour_condition(tilt, d_xy, above, geom["cup_radius_m"])
        if now and first is None:
            first = step
        if dwell.update(now) and fired is None:
            fired = step
        if step % 20 == 0 or step in (first, fired):
            print(f"   step {step:3d}  tilt {tilt:5.1f} deg  mouth-to-cup {100 * d_xy:5.1f} cm  "
                  f"above rim {100 * above:5.1f} cm  {'over the cup' if now else ''}")
    print(f"   criterion first met at step {first}, success declared at step {fired} "
          f"(dwell {dwell.dwell}); longest hold {dwell.best} steps\n")

    print("3. physics gate on the shipped phase-3 replay reports (lift >= 6 cm, drift <= 5 cm, cup <= 2 cm)")
    reports = sorted((ROOT / "results" / "phase3_rollouts").glob("cond000_ep*.json"))
    for path in reports:
        r = json.loads(path.read_text())
        print(f"   {path.stem}: lift {100 * r['bottle_lift']:.1f} cm, drift "
              f"{100 * r['bottle_vs_reference']['gap_max_m']:.1f} cm, cup moved "
              f"{100 * r['cup_moved_m']:.3f} cm, rubric progress {r['progress_max']:.2f} "
              f"-> {'kept' if passes_physics_gate(r) else 'dropped'}")
    r = copy.deepcopy(json.loads(reports[0].read_text()))
    r["bottle_lift"], r["cup_moved_m"] = 0.012, 0.082
    print(f"   synthetic failure: dropped because {'; '.join(physics_gate_reasons(r))}")


if __name__ == "__main__":
    main()
