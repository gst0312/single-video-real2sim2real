"""Put an opaque, cloth-coloured backing sheet just under the tabletop cloth.

The cloth layer is semi-transparent, and what shows through at the deployment camera's
grazing angle is stacked dark junk floating under and behind the table - that is the
mottling on the rendered tabletop. Recolouring the cloth barely moves the image (the
bleed dominates), solidifying the cloth surfaces its fringe noise, and killing the junk
just exposes deeper junk. Blocking works: a dense near-opaque sheet under the cloth gives
the translucency a clean uniform backdrop, and the real cloth texture stays on top.

The sheet follows the cloth: its cells take the local median cloth height minus
`--depth`, and cells with no cloth above them are skipped, so the sheet clips itself to
the real tabletop outline and tilt. Interior holes are closed morphologically without
letting the outline creep. Colour starts from the local median cloth colour (`--shade`
darkens it); the real fit happens afterwards in `tune_backing_colors.py`.
"""

import argparse

import numpy as np
from plyfile import PlyData, PlyElement

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True)
p.add_argument("--table-z", type=float, default=-0.020)
p.add_argument("--z-band", type=float, nargs=2, default=[-0.045, 0.04])
p.add_argument("--x-range", type=float, nargs=2, default=[-0.30, 0.95])
p.add_argument("--y-range", type=float, nargs=2, default=[-0.75, 0.75])
p.add_argument("--cell", type=float, default=0.06, help="footprint/colour cell size")
p.add_argument("--spacing", type=float, default=0.015, help="backing gaussian grid")
p.add_argument("--depth", type=float, default=0.025, help="metres below the local cloth")
p.add_argument("--disc", type=float, default=0.012, help="backing gaussian radius")
p.add_argument("--alpha", type=float, default=0.97)
p.add_argument("--shade", type=float, default=0.7,
               help="multiply the starting backing colour; <1 darkens it")
p.add_argument("--min-cloth", type=int, default=2,
               help="a cell needs this many cloth gaussians above it to get backing")
p.add_argument("--out", required=True)
args = p.parse_args()

C0 = 0.28209479177387814

ply = PlyData.read(args.splat)
data = ply["vertex"].data
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
alpha = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
lum = (dc * C0 + 0.5).mean(1)

cloth = (np.abs(xyz[:, 2] - args.table_z - np.mean(args.z_band)) <
         (args.z_band[1] - args.z_band[0]) / 2) & \
        (xyz[:, 0] > args.x_range[0]) & (xyz[:, 0] < args.x_range[1]) & \
        (xyz[:, 1] > args.y_range[0]) & (xyz[:, 1] < args.y_range[1]) & \
        (alpha > 0.08) & (lum > 0.06) & (lum < 0.7)

nx = int(np.ceil((args.x_range[1] - args.x_range[0]) / args.cell))
ny = int(np.ceil((args.y_range[1] - args.y_range[0]) / args.cell))
ix = np.clip(((xyz[cloth, 0] - args.x_range[0]) / args.cell).astype(int), 0, nx - 1)
iy = np.clip(((xyz[cloth, 1] - args.y_range[0]) / args.cell).astype(int), 0, ny - 1)
count = np.zeros((nx, ny), int)
np.add.at(count, (ix, iy), 1)
zsum = np.zeros((nx, ny))
np.add.at(zsum, (ix, iy), xyz[cloth, 2])
csum = np.zeros((nx, ny, 3))
np.add.at(csum, (ix, iy), dc[cloth])
ok = count >= args.min_cloth
zmed = np.where(ok, zsum / np.maximum(count, 1), np.nan)
cmed = np.where(ok[..., None], csum / np.maximum(count, 1)[..., None], np.nan)
print(f"{int(cloth.sum())} cloth gaussians, {int(ok.sum())}/{nx * ny} cells with cloth")

# close interior holes without letting the outline creep: morphological close (dilate
# then erode by the same amount) defines the sheet's footprint, and values inside it are
# filled iteratively from valid neighbours. Shifts are zero-padded, never wrapped.


def shift(a, dx, dy, fill=0):
    out = np.full_like(a, fill)
    xs_src = slice(max(0, -dx), a.shape[0] - max(0, dx))
    ys_src = slice(max(0, -dy), a.shape[1] - max(0, dy))
    xs_dst = slice(max(0, dx), a.shape[0] - max(0, -dx))
    ys_dst = slice(max(0, dy), a.shape[1] - max(0, -dy))
    out[xs_dst, ys_dst] = a[xs_src, ys_src]
    return out


def neighbours(a, fill=0):
    return [shift(a, dx, dy, fill) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)]


footprint = ok.copy()
for _ in range(3):   # dilate
    footprint = footprint | np.any(np.stack(neighbours(footprint)), axis=0)
for _ in range(3):   # erode by the same amount
    footprint = footprint & np.all(np.stack(neighbours(footprint, fill=True)), axis=0)
footprint |= ok

for _ in range(8):
    valid = ~np.isnan(zmed)
    nb_v = np.stack(neighbours(valid))
    nb_z = np.stack(neighbours(np.nan_to_num(zmed)))
    nb_c = np.stack([np.stack(neighbours(np.nan_to_num(cmed[..., i]))) for i in range(3)], -1)
    cnt = nb_v.sum(0)
    fill = (~valid) & footprint & (cnt >= 2)
    zmed[fill] = (nb_z.sum(0))[fill] / cnt[fill]
    cmed[fill] = (nb_c.sum(0))[fill] / cnt[fill][..., None]
ok = ~np.isnan(zmed) & footprint
print(f"after hole filling: {int(ok.sum())}/{nx * ny} cells with backing")

pts, cols = [], []
gx = np.arange(args.x_range[0], args.x_range[1], args.spacing)
gy = np.arange(args.y_range[0], args.y_range[1], args.spacing)
for x in gx:
    cx_i = min(int((x - args.x_range[0]) / args.cell), nx - 1)
    for y in gy:
        cy_i = min(int((y - args.y_range[0]) / args.cell), ny - 1)
        if not ok[cx_i, cy_i]:
            continue
        pts.append((x, y, zmed[cx_i, cy_i] - args.depth))
        cols.append(cmed[cx_i, cy_i])
pts = np.array(pts)
cols = np.array(cols)
cols = ((cols * C0 + 0.5) * args.shade - 0.5) / C0

rows = np.zeros(len(pts), dtype=data.dtype)
rows["x"], rows["y"], rows["z"] = pts.T
for i in range(3):
    rows[f"f_dc_{i}"] = cols[:, i]
for i in range(3):
    rows[f"scale_{i}"] = np.log(args.disc)
rows["rot_0"] = 1.0
rows["opacity"] = np.log(args.alpha / (1 - args.alpha))
out = np.concatenate([data, rows])
PlyData([PlyElement.describe(out, "vertex")]).write(args.out)
print(f"added {len(rows)} backing gaussians -> {len(out)}; wrote {args.out}")
