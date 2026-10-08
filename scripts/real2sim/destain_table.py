"""Flatten the tabletop's baked stains against the real frame: a low-frequency,
per-gaussian photometric correction.

The global relight (relight_splat.py) fits one gain/offset for the whole scene; the
reconstruction additionally bakes a low-frequency stain field into the cloth (visible as
blotches once the deployment exposure brightens the table; GSWorld's own render carries
the same field at their darker exposure). The real frame is available pixel-aligned, so
the correction is measured, not invented: smooth the real and the rendered table with a
wide gaussian, take their ratio per channel, and multiply every table-band gaussian's
colour by the ratio sampled at its projected pixel. Same fit-against-the-real-frame
mechanism as the relight, one step lower in spatial order; robot and curtain pixels are
masked out of the fit, and the ratio is clamped so the correction stays photometric.

    python scripts/real2sim/destain_table.py --splat in.ply --real real.png --sim sim.png \
        --pose zed_pose_fk.npz --out out.ply
"""

import argparse
import os

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt, gaussian_filter

C0 = 0.28209479177387814


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--sim", required=True, help="render of --splat from --pose")
    p.add_argument("--pose", required=True)
    p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
    p.add_argument("--sigma", type=float, default=25.0, help="field smoothing, px")
    p.add_argument("--clamp", type=float, nargs=2, default=[0.82, 1.22])
    p.add_argument("--table-band", type=float, nargs=6,
                   default=[-0.30, 0.82, -0.75, 0.75, -0.07, 0.03],
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="world box of gaussians the correction applies to")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import json
    real = np.asarray(Image.open(args.real).convert("RGB"), np.float64)
    sim = np.asarray(Image.open(args.sim).convert("RGB"), np.float64)
    h, w = real.shape[:2]

    # fit mask: pixels that are tabletop in both images - grey, not the white robot,
    # not the dark curtain/floor
    lum_r, lum_s = real.mean(2), sim.mean(2)
    mask = (lum_r > 70) & (lum_r < 200) & (lum_s > 70) & (lum_s < 200)
    mask[:340] = False          # everything above the far table edge
    ratio = np.ones((h, w, 3))
    m = mask.astype(np.float64)
    for c in range(3):
        num = gaussian_filter(real[:, :, c] * m, args.sigma)
        den = gaussian_filter(sim[:, :, c] * m, args.sigma)
        cover = gaussian_filter(m, args.sigma)
        ok = cover > 0.25
        ratio[:, :, c][ok] = (num[ok] / np.maximum(den[ok], 1e-3))
    # extend the field into masked-out pixels from the nearest fitted pixel
    okpx = gaussian_filter(m, args.sigma) > 0.25
    idx = distance_transform_edt(~okpx, return_distances=False, return_indices=True)
    ratio = ratio[idx[0], idx[1]]
    ratio = np.clip(ratio, args.clamp[0], args.clamp[1])

    intr = json.load(open(args.intrinsics))
    rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
    fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]
    pose = np.load(args.pose)
    cpos, crot = pose["pos"].astype(np.float64), pose["rot"].astype(np.float64)

    ply = PlyData.read(args.splat)
    data = ply["vertex"].data.copy()
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    x0, x1, y0, y1, z0, z1 = args.table_band
    band = ((xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
            & (xyz[:, 2] > z0) & (xyz[:, 2] < z1))
    cam = (xyz[band] - cpos) @ crot          # world -> camera (columns are axes)
    z = cam[:, 2]
    u = np.clip(np.round(cam[:, 0] / z * fx + cx).astype(int), 0, w - 1)
    v = np.clip(np.round(cam[:, 1] / z * fy + cy).astype(int), 0, h - 1)
    g = ratio[v, u]                          # (n, 3)
    vis = z > 0.1
    print(f"table band {int(band.sum())} gaussians, correcting {int(vis.sum())}; "
          f"field {np.percentile(g[vis], [5, 50, 95], axis=0).round(3).tolist()}")

    sel = np.nonzero(band)[0][vis]
    for c in range(3):
        gc = g[vis, c]
        col = data[f"f_dc_{c}"][sel].astype(np.float64) * C0 + 0.5
        data[f"f_dc_{c}"][sel] = ((col * gc - 0.5) / C0).astype(np.float32)
        for k in range(15):
            key = f"f_rest_{c * 15 + k}"
            data[key][sel] = (data[key][sel].astype(np.float64) * gc).astype(np.float32)

    PlyData([PlyElement.describe(data, "vertex")]).write(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
