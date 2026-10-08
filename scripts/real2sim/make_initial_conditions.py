"""Write initial_conditions.json: the language instruction plus N object layouts.

Same file PolaRiS's load_eval_initial_conditions reads, same shape as the official
environments: {"instruction": ..., "poses": [{object: [x, y, z, qw, qx, qy, qz]}, ...]}.

The layouts are PERTURBATIONS OF THE DEMONSTRATION'S OWN LAYOUT, not independent draws
over the table. This is the one thing that has to be right, and the first version got it
wrong (2026-08-12 to 2026-08-13): it sampled the bottle and the cup independently and
uniformly over x 0.36..0.60, y +-0.24, keeping only a 14 cm separation, so about half the
layouts put the cup on the opposite side of the bottle from where the demonstration had it.
The synthesiser then had to yaw the whole demonstrated path by a large angle to reach that
cup, which swings the pour into places the demonstration never visited - measured on the
first physics rollout: the bottle swept through the cup and shoved it 8.2 cm. That same bad
layout is also what made trajgen's own `traj_interp` look unusable (endpoint displacements
of tens of centimetres with sign flips); it is not a property of the task.

Both references perturb around the demonstration instead:

- r2r2r `franka_coffee_maker.py` randomises the grasped object by
  `(rand*2-1) * 0.06` about its demo pose and generates new starts with
  `generate_directional_starts(magnitude=0.1, direction_weight=0.7, perp_variation=0.10)`,
  i.e. biased along the demonstrated direction.
- LFHV's last round is stricter and is what this follows (user's call, 2026-08-13):
  "Only the mustard bottle is randomised, matching eval, so the cup stays at its demo"
  (`tools/datagen/gen_kinematic_states.py:66`), with the bottle drawn
  `(rand*2-1) * 0.05` in xy and `+-10 degrees` of yaw (their L419-425; the file's own
  header comment still says the older +-0.06 / +-pi/8, the code is the authority).

Both object meshes were scanned with their long axis along +Y and their base at y = 0, so
standing one upright is a +90 degree rotation about x; after that the base sits exactly at
the prim origin, and the placement height is simply the table surface. The demonstration's
own yaw is recovered from the tracked pose and kept as the centre of the bottle's draw, so
the label faces where it actually faced.
"""

import argparse
import os
import json

import numpy as np

UPRIGHT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])  # +90 deg about x, (w, x, y, z)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ])


def yaw_quat(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def rotate(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return 2 * u.dot(v) * u + (w * w - u.dot(u)) * v + 2 * w * np.cross(u, v)


def demo_layout(track_path):
    """The demonstration's own layout: (bottle xy, bottle yaw, cup xy, cup yaw).

    take4.npy holds both objects' tracked poses in the base frame. The quaternion lane is
    identified the way `synthesize_trajectories.detect_quat_order` does - the cup never
    leaves the table, so the reading that keeps its mesh +Y along world +-Z is the right
    one. Yaw is read off the mesh's +Z axis (the bottle's label side) projected into the
    table plane, so the upright convention above can be rebuilt exactly.
    """
    track = np.load(track_path)
    for order in ([0, 1, 2, 3], [3, 0, 1, 2]):
        q = track[1, :, 3:][:, order]
        up_z = np.array([rotate(qi, np.array([0.0, 1.0, 0.0]))[2] for qi in q])
        if np.abs(up_z).min() > 0.95:
            break
    else:
        raise SystemExit("neither quaternion order keeps the cup upright")

    def spin(q):
        """The z rotation to apply ON TOP OF `UPRIGHT` to reproduce q's heading.

        `UPRIGHT` alone already carries the mesh's +Z (the label side) to world -Y, so the
        parameter this file uses is offset from the label's world bearing by that -90
        degrees. Measuring the bearing and subtracting UPRIGHT's own keeps the two
        conventions from drifting apart - they did once: the label bearing of the
        demonstration is -3.3 degrees, which is this parameter at +86.7, and the previous
        default of +90 +- 20 was solved independently from the mesh texture on 2026-08-12.
        The two agree to 3 degrees, which is the check that both readings are right.
        """
        face = rotate(q, np.array([0.0, 0.0, 1.0]))
        ref = rotate(UPRIGHT, np.array([0.0, 0.0, 1.0]))
        return float(np.arctan2(face[1], face[0]) - np.arctan2(ref[1], ref[0]))

    bottle_q = track[0, 0, 3:][order]
    cup_q = track[1, 0, 3:][order]
    return (track[0, 0, :3][:2], spin(bottle_q),
            track[1, :, :3].mean(0)[:2], spin(cup_q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="pour the mustard into the blue cup")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track",
                   default=os.environ.get("DEMO_TRACK", "take4.npy"),
                   help="the tracked demonstration whose layout the draws are centred on")
    p.add_argument("--table-z", type=float, default=-0.019,
                   help="where an object's base sits; the collision platform's top face is "
                        "at -0.020, so this leaves a millimetre to settle through instead "
                        "of spawning inside the surface")
    p.add_argument("--bottle-xy", type=float, default=0.05,
                   help="half-width of the bottle's uniform xy draw about the demo pose "
                        "(LFHV's last round; r2r2r uses 0.06)")
    p.add_argument("--bottle-yaw-deg", type=float, default=10.0,
                   help="half-width of the bottle's uniform yaw draw about the demo yaw")
    p.add_argument("--cup-xy", type=float, default=0.03,
                   help="same for the cup. LFHV's last round used 0 - the cup never moves - "
                        "but a policy trained on that learns the pour target as a fixed "
                        "place in the workspace instead of looking for it (2026-08-13 user "
                        "decision), so the cup gets a small draw of its own. Small on "
                        "purpose: the pour is retargeted onto it, and that only behaves "
                        "while the displacement stays in the interpolation's regime")
    p.add_argument("--cup-yaw-deg", type=float, default=15.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    b_xy0, b_yaw0, c_xy0, c_yaw0 = demo_layout(args.track)
    print(f"demonstration layout: bottle {np.round(b_xy0, 3)} yaw {np.degrees(b_yaw0):+.1f} deg, "
          f"cup {np.round(c_xy0, 3)} yaw {np.degrees(c_yaw0):+.1f} deg "
          f"(yaw is the spin on top of UPRIGHT; the bottle's +90 default solved from the "
          f"mesh texture on 2026-08-12 agrees to "
          f"{abs(np.degrees(b_yaw0) - 90):.1f} deg)")
    print(f"  cup is {np.round(c_xy0 - b_xy0, 3)} m from the bottle "
          f"({np.linalg.norm(c_xy0 - b_xy0):.3f} m, bearing "
          f"{np.degrees(np.arctan2(*(c_xy0 - b_xy0)[::-1])):+.1f} deg)")

    poses = []
    for _ in range(args.count):
        b_xy = b_xy0 + rng.uniform(-args.bottle_xy, args.bottle_xy, size=2)
        c_xy = c_xy0 + rng.uniform(-args.cup_xy, args.cup_xy, size=2) if args.cup_xy else c_xy0
        b_yaw = b_yaw0 + rng.uniform(-1, 1) * np.radians(args.bottle_yaw_deg)
        c_yaw = c_yaw0 + rng.uniform(-1, 1) * np.radians(args.cup_yaw_deg)
        poses.append({
            "mustard": [float(b_xy[0]), float(b_xy[1]), args.table_z,
                        *[float(v) for v in quat_mul(yaw_quat(b_yaw), UPRIGHT)]],
            "blue_cup": [float(c_xy[0]), float(c_xy[1]), args.table_z,
                         *[float(v) for v in quat_mul(yaw_quat(c_yaw), UPRIGHT)]],
        })

    with open(args.out, "w") as f:
        json.dump({"instruction": args.instruction, "poses": poses}, f, indent=1)
    sep = [np.linalg.norm(np.array(q["mustard"][:2]) - np.array(q["blue_cup"][:2])) for q in poses]
    print(f"wrote {args.out}: {len(poses)} layouts")
    print(f"  instruction: {args.instruction!r}")
    print(f"  bottle xy +-{args.bottle_xy} m, yaw +-{args.bottle_yaw_deg} deg; "
          f"cup xy +-{args.cup_xy} m, yaw +-{args.cup_yaw_deg} deg; table z {args.table_z}")
    print(f"  bottle-to-cup distance {min(sep):.3f} to {max(sep):.3f} m")


if __name__ == "__main__":
    main()
