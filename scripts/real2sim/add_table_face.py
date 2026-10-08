"""Give the table a front face: a curtain of small matte gaussians on the front plane,
coloured from the real frame.

The reconstruction has almost no gaussians on the table's front face (61 in the whole
plane) - the pixels there are painted by the lower tails of the big rim gaussians, so
they render bright and structureless, and no region-based photometric edit can darken
them without dimming the rim above (same gaussians). The face the real camera sees is a
matte cloth plane, so it is added as geometry: a grid of small opaque gaussians on the
front plane, each taking its colour from the real frame at its projected pixel (the same
measured-from-the-real-frame principle as destain_table.py), zero SH. They occlude the
tails, and the real edge structure - bright rim, darker face - appears.

    python scripts/real2sim/add_table_face.py --splat in.ply --real real.png \
        --pose zed_pose_fk.npz --out out.ply
"""

import argparse
import os
import json

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
    p.add_argument("--face-x", type=float, default=0.795)
    p.add_argument("--y-range", type=float, nargs=2, default=[-0.62, 0.62])
    p.add_argument("--z-range", type=float, nargs=2, default=[-0.165, -0.032])
    p.add_argument("--spacing", type=float, default=0.006)
    p.add_argument("--scale", type=float, default=0.005, help="gaussian radius, metres")
    p.add_argument("--semantics", default=None)
    p.add_argument("--semantics-out", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    real = np.asarray(Image.open(args.real).convert("RGB"), np.float64) / 255.0
    h, w = real.shape[:2]
    intr = json.load(open(args.intrinsics))
    rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
    fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]
    pose = np.load(args.pose)
    cpos, crot = pose["pos"].astype(np.float64), pose["rot"].astype(np.float64)

    ys = np.arange(args.y_range[0], args.y_range[1], args.spacing)
    zs = np.arange(args.z_range[0], args.z_range[1], args.spacing)
    gy, gz = np.meshgrid(ys, zs)
    n = gy.size
    rng = np.random.default_rng(0)
    xyz = np.stack([np.full(n, args.face_x) + rng.normal(0, 0.001, n),
                    gy.ravel() + rng.normal(0, 0.001, n),
                    gz.ravel() + rng.normal(0, 0.001, n)], 1)

    cam = (xyz - cpos) @ crot
    z = cam[:, 2]
    u = np.clip(np.round(cam[:, 0] / z * fx + cx).astype(int), 0, w - 1)
    v = np.clip(np.round(cam[:, 1] / z * fy + cy).astype(int), 0, h - 1)
    col = real[v, u]

    ply = PlyData.read(args.splat)
    data = ply["vertex"].data
    row = np.zeros(n, dtype=data.dtype)
    row["x"], row["y"], row["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for c in range(3):
        row[f"f_dc_{c}"] = (col[:, c] - 0.5) / C0
    row["opacity"] = 3.0                     # sigmoid(3) = 0.95, a solid skin
    for i in range(3):
        row[f"scale_{i}"] = np.log(args.scale)
    row["rot_0"] = 1.0
    if "nx" in data.dtype.names:
        pass                                 # normals stay zero
    out = np.concatenate([data, row])
    PlyData([PlyElement.describe(out, "vertex")]).write(args.out)
    print(f"added {n} face gaussians at x={args.face_x}; wrote {args.out} ({len(out)})")
    if args.semantics and args.semantics_out:
        sem = np.load(args.semantics)
        np.save(args.semantics_out, np.concatenate([sem, np.full(n, -1, dtype=sem.dtype)]))
        print(f"wrote {args.semantics_out}")


if __name__ == "__main__":
    main()
