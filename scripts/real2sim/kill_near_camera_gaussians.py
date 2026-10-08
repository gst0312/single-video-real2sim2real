"""Delete gaussians sitting between the deployment camera and the scene.

The camera was standing on a tripod when the scene was photographed, so the reconstruction
contains the camera and its tripod, a few centimetres in front of where the deployment view
is rendered from. Whatever sits there is an artefact of that, and it veils the bottom of
every frame.

How deep to cut is set by the table's front edge, which is about 0.5 m away: measured
against a real frame over the strip the cut touches, 0.25 m and 0.45 m give the same picture
(mean absolute error over that strip 9.33 -> 7.89 after fitting exposure), 0.60 m starts to
eat the edge (8.41, and the gradient correlation drops from 0.58 to 0.55) and 0.75 m removes
the cloth's dark backing and wrecks it (21.75, 0.39).

Rows are deleted, so any per-gaussian labels have to be filtered with them; pass
`--semantics` and the matching file is written next to the output.

`--fog-box` extends the same cleanup to the gap the depth cone cannot reach: the cone
stops at 0.4 m because 0.6 m starts eating the table edge, which leaves the air between
the cone and the table's front edge (x 0.82..1.15 in the base frame) holding a few
hundred overexposed floaters. They project exactly onto the table's front face in the
deployment view and read as a bright band along the near edge.
"""

import argparse

import numpy as np
from plyfile import PlyData, PlyElement

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True)
p.add_argument("--pose", required=True)
p.add_argument("--max-depth", type=float, default=0.4, help="kill in front of the camera out to here")
p.add_argument("--fog-box", type=float, nargs=6, action="append", default=None,
               metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
               help="also kill everything inside this base-frame box; repeatable. Used "
                    "for the air between the depth cone and the table's front edge, the "
                    "overhang drape, and the scene-labelled wrist-hardware crumbs left "
                    "hanging at the scan pose (the moving dark dot in wrist views)")
p.add_argument("--semantics", default=None, help="per-gaussian labels to filter alongside")
p.add_argument("--out", required=True)
args = p.parse_args()

pose = np.load(args.pose)
c, rot = pose["pos"], pose["rot"]
axis = rot[:, 2]

ply = PlyData.read(args.splat)
vert = ply["vertex"]
xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1)
d = xyz - c
depth = d @ axis
lateral = np.linalg.norm(d - depth[:, None] * axis, axis=1)
kill = (depth > 0) & (depth < args.max_depth) & (lateral < depth * 1.3 + 0.25)
for fb in (args.fog_box or []):
    x0, x1, y0, y1, z0, z1 = fb
    box = ((xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
           & (xyz[:, 2] > z0) & (xyz[:, 2] < z1))
    print(f"fog box {fb}: {int(box.sum())} gaussians")
    kill |= box
print(f"{len(xyz)} gaussians, killing {int(kill.sum())} within {args.max_depth} m of the camera")
PlyData([PlyElement.describe(vert.data[~kill], "vertex")]).write(args.out)
print(f"wrote {args.out} with {int((~kill).sum())} gaussians")
if args.semantics:
    sem = np.load(args.semantics)
    if len(sem) != len(xyz):
        raise SystemExit(f"{len(sem)} labels for {len(xyz)} gaussians")
    out = args.out.rsplit(".", 1)[0] + "_semantics.npy"
    np.save(out, sem[~kill])
    print(f"wrote {out}")
