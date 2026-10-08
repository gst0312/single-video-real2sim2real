"""Hide gaussians hovering over the tabletop, where nothing physical exists.

GSWorld's own paint-out does this (final_paintout.json, `overtable_kill`, d in
0.13..0.9 over the table area, arm excluded) but with `surface_lum_min: 0.35` it
only kills bright glow. The dark floaters it spares hang over the dark cloth and
smear into large blotches at the deployment camera's grazing angle. This is the
same kill completed: any luminance, and starting just above the reconstruction's
own surface noise (the census shows the real surface lives within +-2 cm and the
+5..+13 cm band is already empty, so a +3 cm floor cuts nothing real).

Empty-scene asset only: objects come in later as raytraced meshes, and the robot
volumes are excluded. Hides by opacity (-15) instead of deleting rows, so the
per-gaussian semantics stay aligned.
"""

import argparse

import numpy as np
from plyfile import PlyData, PlyElement

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True, help="scene splat in the robot base frame")
p.add_argument("--out", required=True)
p.add_argument("--table-z", type=float, default=-0.045, help="tabletop height, base frame")
p.add_argument("--h-min", type=float, default=0.03, help="kill from this high above the table")
p.add_argument("--h-max", type=float, default=1.2)
p.add_argument("--table-x", type=float, nargs=2, default=[-0.25, 0.85])
p.add_argument("--table-y", type=float, nargs=2, default=[-0.55, 0.75])
p.add_argument("--arm-centres", type=float, nargs="*", default=[0.0, 0.0, 0.05, 0.72],
               help="x y pairs of robot bases to exclude")
p.add_argument("--arm-radius", type=float, default=0.28)
p.add_argument("--exclude-centres", type=float, nargs="*",
               default=[0.2776, -0.2079, 0.0997, 0.2087],
               help="x y pairs to exempt below --exclude-height: the restored ArUco "
                    "surfaces reach a few cm up and must not be re-killed")
p.add_argument("--exclude-radius", type=float, default=0.075)
p.add_argument("--exclude-height", type=float, default=0.09)
p.add_argument("--semantics", default=None,
               help="per-gaussian labels; only background (-1) is killed, so the labelled "
                    "robot survives for the split no matter how far it leans off its base")
args = p.parse_args()

ply = PlyData.read(args.splat)
vert = ply["vertex"]
data = vert.data.copy()
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)

h = xyz[:, 2] - args.table_z
over = (h > args.h_min) & (h < args.h_max) & \
       (xyz[:, 0] > args.table_x[0]) & (xyz[:, 0] < args.table_x[1]) & \
       (xyz[:, 1] > args.table_y[0]) & (xyz[:, 1] < args.table_y[1])
centres = np.array(args.arm_centres, dtype=np.float64).reshape(-1, 2)
for c in centres:
    over &= np.linalg.norm(xyz[:, :2] - c, axis=1) > args.arm_radius
for c in np.array(args.exclude_centres, dtype=np.float64).reshape(-1, 2):
    over &= (np.linalg.norm(xyz[:, :2] - c, axis=1) > args.exclude_radius) | \
            (h > args.exclude_height)
if args.semantics:
    sem = np.load(args.semantics)
    assert len(sem) == len(data)
    over &= sem < 0

sig = 1 / (1 + np.exp(-data["opacity"][over].astype(np.float64)))
print(f"{len(data)} gaussians; hiding {int(over.sum())} over the table "
      f"(visible mass {sig.sum():.0f}) between {args.h_min} and {args.h_max} m up")
data["opacity"][over] = -15.0
PlyData([PlyElement.describe(data, "vertex")]).write(args.out)
print(f"wrote {args.out}")
