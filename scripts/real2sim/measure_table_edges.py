"""Measure the table-edge offsets between a real frame and a render, edge by edge.

For sample points along each named edge segment the luminance gradient is probed along
the edge normal in both images; the strongest peak (parabola-refined to sub-pixel) marks
the edge, and the per-edge median of (sim - real) says how far and which way that edge is
off. This is the acceptance number for "table-edge alignment" - region-shift metrics (the earlier
+6/-8 px record) mix the edge with everything else in the box, and a 50% blend only shows
misalignment above a couple of pixels.

Segments are given in the 1280x720 deployment frame, chosen off the robots and plates;
the same segments work for every frame of the fixed external camera. GSWorld's published
render measures -1.3/+0.2 px on the left/near edges against the same real frame, which is
the parity bar.

    python scripts/real2sim/measure_table_edges.py --real real.png --sim sim.png --label after-fix
"""

import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SEGMENTS = {
    # (x0, y0, x1, y1), endpoints on the edge, probes run along the normal
    "left_edge": (513, 362, 404, 672),
    "near_edge": (432, 684, 900, 706),
    "far_edge_l": (527, 361, 585, 364),
    "far_edge_r": (778, 368, 930, 373),
}
SEARCH = 16
STEP = 12


def lum(path):
    return gaussian_filter(
        np.asarray(Image.open(path).convert("L")).astype(np.float64), 1.2)


def peak_offset(img, p, n):
    ts = np.arange(-SEARCH, SEARCH + 1, 1.0)
    xs, ys = p[0] + n[0] * ts, p[1] + n[1] * ts
    if xs.min() < 1 or xs.max() > 1278 or ys.min() < 1 or ys.max() > 718:
        return None
    prof = img[np.round(ys).astype(int), np.round(xs).astype(int)]
    g = np.abs(np.gradient(prof))
    k = int(np.argmax(g[2:-2])) + 2
    if g[k] < 2.0:
        return None
    denom = g[k - 1] - 2 * g[k] + g[k + 1]
    sub = 0.5 * (g[k - 1] - g[k + 1]) / denom if abs(denom) > 1e-9 else 0.0
    return ts[k] + np.clip(sub, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    real, sim = lum(args.real), lum(args.sim)

    print(f"== {args.label or args.sim}")
    for name, (x0, y0, x1, y1) in SEGMENTS.items():
        d = np.array([x1 - x0, y1 - y0], float)
        L = np.linalg.norm(d)
        d /= L
        n = np.array([-d[1], d[0]])
        offs = []
        for t in np.arange(0, L, STEP):
            p = np.array([x0, y0]) + d * t
            a = peak_offset(real, p, n)
            b = peak_offset(sim, p, n)
            if a is not None and b is not None and abs(a) < SEARCH - 2 and abs(b) < SEARCH - 2:
                offs.append(b - a)
        offs = np.array(offs)
        if len(offs) < 4:
            print(f"  {name:11s}: too few samples ({len(offs)})")
            continue
        q1, q2, q3 = np.percentile(offs, [25, 50, 75])
        print(f"  {name:11s}: median {q2:+.1f} px  (IQR {q1:+.1f}..{q3:+.1f}, n={len(offs)})")


if __name__ == "__main__":
    main()
