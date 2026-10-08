"""One closed-loop step on the backing sheet's colours, against a real frame.

A uniform backing under semi-transparent cloth still renders uneven: the cloth's own
transmittance varies, so the mix does. The backing rows carry no real texture - they are
free parameters - so their colours can simply be fitted: render, take the low-pass
difference to the real frame over the tabletop, and add it (scaled) to each backing
gaussian's colour where it projects. Two or three rounds with quick_render_scene.py
converge the tabletop's low-frequency field onto the real one; the cloth's
high-frequency texture rides on top untouched.

The fitting frame must have the arm parked away (home_static). Bright pixels (plates,
arm) and pixels with strong local gradient in the real frame (the printed markers, which
the marker-free sim does not have, plus cables and plate edges) are excluded.

Driven by a loop of quick_render_scene.py + this script; --n-backing is how many rows at
the tail of the ply are the sheet.
"""

import argparse
import os
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from plyfile import PlyData, PlyElement

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True, help="ply whose tail rows are the backing sheet")
p.add_argument("--n-backing", type=int, required=True)
p.add_argument("--real", required=True)
p.add_argument("--sim", required=True)
p.add_argument("--pose", default=os.path.join(os.environ.get("WORK", "work"), "zed_pose_fk.npz"))
p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
p.add_argument("--pixel-lum-max", type=float, default=150.0)
p.add_argument("--grad-max", type=float, default=6.0,
               help="exclude pixels whose 9x9 mean gradient in the real frame exceeds this")
p.add_argument("--lowpass", type=int, default=18)
p.add_argument("--rate", type=float, default=1.3, help="update step on the difference")
p.add_argument("--out", required=True)
args = p.parse_args()

C0 = 0.28209479177387814

real = np.asarray(Image.open(args.real).convert("RGB")).astype(np.float64)
sim = np.asarray(Image.open(args.sim).convert("RGB")).astype(np.float64)
h, w = real.shape[:2]
intr = json.load(open(args.intrinsics))
rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]
pose = np.load(args.pose)
rot, pos = pose["rot"].astype(np.float64), pose["pos"].astype(np.float64)

ply = PlyData.read(args.splat)
data = ply["vertex"].data.copy()
back = data[-args.n_backing:]
xyz = np.stack([back["x"], back["y"], back["z"]], 1).astype(np.float64)

cam = (xyz - pos) @ rot
z = cam[:, 2]
u = cam[:, 0] / z * fx + cx
v = cam[:, 1] / z * fy + cy
ok = (z > 0.2) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)

diff = real - sim
still = (real.mean(2) < args.pixel_lum_max) & (sim.mean(2) < args.pixel_lum_max)
gy, gx = np.gradient(real.mean(2))
grad = np.hypot(gy, gx)
pad = np.pad(grad, 4, mode="edge")
grad_local = sliding_window_view(pad, (9, 9)).mean(axis=(2, 3))
still &= grad_local < args.grad_max


def box(a, k):
    csum = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0))), 0), 1)
    hh, ww = a.shape
    y0 = np.clip(np.arange(hh) - k, 0, hh)
    y1 = np.clip(np.arange(hh) + k + 1, 0, hh)
    x0 = np.clip(np.arange(ww) - k, 0, ww)
    x1 = np.clip(np.arange(ww) + k + 1, 0, ww)
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (csum[y1][:, x1] - csum[y1][:, x0] - csum[y0][:, x1] + csum[y0][:, x0]) / area


w_mask = box(still.astype(np.float64), args.lowpass)
for i in range(3):
    d = box(np.where(still, diff[..., i], 0.0), args.lowpass) / np.maximum(w_mask, 1e-3)
    d = np.clip(d, -60, 60)
    samp = d[v[ok].astype(int), u[ok].astype(int)] / 255.0 * args.rate
    idx = len(data) - args.n_backing + np.nonzero(ok)[0]
    data[f"f_dc_{i}"][idx] = np.clip(
        data[f"f_dc_{i}"][idx] + samp / C0, (0.0 - 0.5) / C0, (1.2 - 0.5) / C0)

res = np.abs(np.where(still, diff.mean(2), 0.0))
print(f"mean |low-pass diff| over table: {box(res, args.lowpass).max():.1f} max, "
      f"{res[still].mean():.1f} mean; updated {int(ok.sum())} backing gaussians")
PlyData([PlyElement.describe(data, "vertex")]).write(args.out)
print(f"wrote {args.out}")
