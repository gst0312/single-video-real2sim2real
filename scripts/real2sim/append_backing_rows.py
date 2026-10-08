"""Append the fitted table-backing sheet to a scene splat, semantics kept aligned.

The sheet's geometry comes from `insert_table_backing.py` (cells follow the cloth) and
its colours from the closed loop in `tune_backing_colors.py` against a real frame; the
result is stored once as a structured-array npy and re-appended deterministically on
every rebuild, like the relight constants.
"""

import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True)
p.add_argument("--semantics", required=True)
p.add_argument("--rows", default=os.path.join(os.environ.get("WORK", "work"), "table_backing_rows.npy"))
p.add_argument("--exposure", type=float, nargs=6, default=None,
               metavar=("GR", "GG", "GB", "BR", "BG", "BB"),
               help="map the stored colours (tuned in the full-relight domain) to another "
                    "exposure: c' = (c - b_full)*(g/g_full) + b, given as gain and offset "
                    "ratios already divided out - pass the per-channel multiplier and the "
                    "additive term in 0..255")
p.add_argument("--out", required=True)
p.add_argument("--semantics-out", required=True)
args = p.parse_args()

ply = PlyData.read(args.splat)
data = ply["vertex"].data
rows = np.load(args.rows).copy()
if args.exposure is not None:
    mult = np.array(args.exposure[:3])
    add = np.array(args.exposure[3:]) / 255.0
    for c in range(3):
        col = rows[f"f_dc_{c}"].astype(np.float64) * C0 + 0.5
        rows[f"f_dc_{c}"] = ((col * mult[c] + add[c] - 0.5) / C0).astype(np.float32)
        for k in range(15):
            key = f"f_rest_{c * 15 + k}"
            if key in rows.dtype.names:
                rows[key] = (rows[key].astype(np.float64) * mult[c]).astype(np.float32)
    print(f"backing colours remapped: x{mult.round(4)} +{(add * 255).round(2)}")
assert rows.dtype == data.dtype, "backing rows dtype does not match the splat"
sem = np.load(args.semantics)
assert len(sem) == len(data), f"{len(sem)} labels for {len(data)} gaussians"

out = np.concatenate([data, rows])
PlyData([PlyElement.describe(out, "vertex")]).write(args.out)
np.save(args.semantics_out, np.concatenate([sem, np.full(len(rows), -1, dtype=sem.dtype)]))
print(f"appended {len(rows)} backing rows -> {len(out)}; wrote {args.out}")
