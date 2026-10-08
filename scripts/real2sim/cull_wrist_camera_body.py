"""Remove the reconstructed wrist camera's own body from the link splats.

The scan captured the ZED mini and its bracket, and the split assigns them to the links
they ride on (panda_link7, gripper base_link). Rendering the wrist view from inside that
reconstruction leaves a dark dot and a veil pinned to the image centre - the camera
photographing itself, which no real frame contains. Gaussians within `--radius` of the
optical centre are hidden (opacity, rows kept); the bracket further out stays, since the
external view really does see it.

Same idea as `kill_near_camera_gaussians.py` for the deployment camera; here the camera
rides a link, so the cut is in the link's own frame and lands in the SEGMENTED plys.
"""

import argparse

import numpy as np
from plyfile import PlyData, PlyElement

# optical centre in each ply's frame: PourMustardCfg's mount in the gripper base_link
# frame, and the same point carried into panda_link7 via link7->link8 (0, 0, 0.107) and
# link8->base_link (0, 0, 0.01817)
CAM = {
    "Gripper-Robotiq_2F_85-base_link-Defeatured_2F_85_PAD_OPEN_basestep_01-"
    "Defeatured_2F_85_PAD_OPEN_basestep.ply": (0.000744, -0.032081, -0.077805),
    "panda_link7-geometry-panda_link7.ply": (0.000744, -0.032081, 0.107 + 0.01817 - 0.077805),
}

#: camera forward axis in the mount frame is -Z (opengl); the mount rotation
#: (PourMustardCfg, wxyz) carries it into the gripper base_link frame, and link7 shares
#: the orientation (fixed link7->link8->base_link chain, translations only)
_q = (-0.402458, 0.587241, 0.565950, -0.415784)
_w, _x, _y, _z = _q
_R = np.array([
    [1 - 2 * _y * _y - 2 * _z * _z, 2 * _x * _y - 2 * _z * _w, 2 * _x * _z + 2 * _y * _w],
    [2 * _x * _y + 2 * _z * _w, 1 - 2 * _x * _x - 2 * _z * _z, 2 * _y * _z - 2 * _x * _w],
    [2 * _x * _z - 2 * _y * _w, 2 * _y * _z + 2 * _x * _w, 1 - 2 * _x * _x - 2 * _y * _y]])
FWD = -_R[:, 2]

C0 = 0.28209479177387814

p = argparse.ArgumentParser()
p.add_argument("--segmented", required=True, help="the SEGMENTED directory, edited in place")
p.add_argument("--radius", type=float, default=0.045)
p.add_argument("--lens-reach", type=float, default=0.30,
               help="also hide DARK gaussians in a cylinder in front of the lens out to "
                    "here: cable/hardware crumbs hanging 5-20 cm ahead of the camera "
                    "render as a slowly drifting dark dot no real frame contains")
p.add_argument("--lens-radius", type=float, default=0.06)
p.add_argument("--lens-lum", type=float, default=0.5,
               help="only rows darker than this leave; the bright gripper stays")
p.add_argument("--island-link", type=float, default=0.02,
               help="single-linkage distance for the debris pass: visible rows cluster "
                    "at this radius, and every cluster except the largest that floats "
                    "further than --island-gap from it is hidden (reconstruction crumbs "
                    "riding a link render as small floating chunks in the wrist view)")
p.add_argument("--island-gap", type=float, default=0.04)
p.add_argument("--island-max", type=int, default=400,
               help="clusters bigger than this are never treated as debris")
args = p.parse_args()

for name, centre in CAM.items():
    path = f"{args.segmented}/{name}"
    ply = PlyData.read(path)
    data = ply["vertex"].data.copy()
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    d = xyz - np.array(centre)
    near = np.linalg.norm(d, axis=1) < args.radius
    t = d @ FWD
    lat = np.linalg.norm(d - t[:, None] * FWD, axis=1)
    lum = (np.stack([data[f"f_dc_{i}"] for i in range(3)], 1) * C0 + 0.5).mean(1)
    ahead = (t > 0.02) & (t < args.lens_reach) & (lat < args.lens_radius) & (lum < args.lens_lum)
    data["opacity"][near | ahead] = -15.0
    PlyData([PlyElement.describe(data, "vertex")]).write(path)
    print(f"{name.split('-')[0]}: hid {int(near.sum())} around the lens and "
          f"{int(ahead.sum())} dark rows ahead of it")

# debris pass over every link ply: voxel connected components; every small component
# floating away from the largest one is reconstruction debris and is hidden
import glob
from scipy import ndimage

for path in sorted(glob.glob(f"{args.segmented}/*.ply")):
    ply = PlyData.read(path)
    data = ply["vertex"].data.copy()
    opa = data["opacity"]
    vis = opa > -3
    if vis.sum() < 100:
        continue
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    v = np.floor(xyz[vis] / args.island_link).astype(int)
    v -= v.min(0)
    grid = np.zeros(v.max(0) + 1, dtype=bool)
    grid[tuple(v.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        continue
    row_lab = lab[tuple(v.T)]
    sizes = np.bincount(row_lab)
    main = int(np.argmax(sizes[1:])) + 1
    body = xyz[vis][row_lab == main]
    hid = 0
    vis_idx = np.nonzero(vis)[0]
    for c in range(1, n + 1):
        if c == main or sizes[c] > args.island_max:
            continue
        rows = vis_idx[row_lab == c]
        centre = xyz[rows].mean(0)
        if np.linalg.norm(body - centre, axis=1).min() > args.island_gap:
            data["opacity"][rows] = -15.0
            hid += len(rows)
    if hid:
        PlyData([PlyElement.describe(data, "vertex")]).write(path)
        print(f"{path.split('/')[-1].split('-')[0]}: hid {hid} debris rows "
              f"({n - 1} satellite clusters checked)")
