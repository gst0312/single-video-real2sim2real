"""Rigid correction from GSWorld's base frame to our marker-derived base frame.

GSWorld anchored their base frame off their own arm fit (Kabsch, 5.3 mm); ours comes from
the two ArUco markers plus the wrist FK chain. The two differ by a small rigid transform
that image residuals against real frames grow with the lever arm (~10 px at the arm's top
and the far table edge, zero at the near edge). This solves that transform once, as a
trimmed iterative-closest-point fit between two clouds of the same physical arm:

  source: the robot gaussians of `fr3_final.ply` (their per-gaussian labels >= 1), taken
          through the inverse of `sim2gs_arm_trans`, i.e. their base frame;
  target: `arm_points_ours.npy`, the arm points of our own COLMAP reconstruction, which
          sits in the marker frame via the two-marker alignment.

Same machinery as legacy/solve_capture_qpos.py --refine-base: nearest neighbour, trim to
the closest fraction, Umeyama with scale held at one, iterate. The result npz feeds
`gsworld_splat_to_2dgs.py --extra-transform` and the camera pose gets composed with it.
"""

import argparse
import os
import json

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polaris_lfhv.alignment import umeyama  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "assets/fr3_robotiq_assets/fr3_final.ply"))
    p.add_argument("--semantics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "assets/fr3_robotiq_assets/fr3_final_semantics_gs.npy"))
    p.add_argument("--config", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "configs/fr3_robotiq_final.json"))
    p.add_argument("--arm-points", default=os.path.join(os.environ.get("WORK", "work"), "arm_points_ours.npy"))
    p.add_argument("--trim", type=float, default=0.6, help="keep this closest fraction, as --refine-base")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    T = np.array(cfg["sim2gs_arm_trans"], np.float64)
    r_cfg, t_cfg = T[:3, :3], T[:3, 3]

    vert = PlyData.read(args.splat)["vertex"].data
    sem = np.load(args.semantics)
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    # visible robot gaussians only, in their base frame
    opa = vert["opacity"].astype(np.float64)
    robot = (sem >= 1) & (opa > -6.0)
    src = (xyz[robot] - t_cfg) @ r_cfg
    tgt = np.load(args.arm_points).astype(np.float64)
    rng = np.random.default_rng(0)
    if len(src) > args.max_points:
        src = src[rng.choice(len(src), args.max_points, replace=False)]
    if len(tgt) > args.max_points:
        tgt = tgt[rng.choice(len(tgt), args.max_points, replace=False)]
    print(f"{len(src)} their arm gaussians vs {len(tgt)} our arm points")

    tree = cKDTree(tgt)
    rot = np.eye(3)
    trans = np.zeros(3)
    for k in range(args.rounds):
        cur = src @ rot.T + trans
        d, idx = tree.query(cur)
        keep = d <= np.quantile(d, args.trim)
        _, rot_new, trans_new = umeyama(src[keep], tgt[idx[keep]], with_scale=False)
        if np.allclose(rot_new, rot, atol=1e-9) and np.allclose(trans_new, trans, atol=1e-9):
            break
        rot, trans = rot_new, trans_new
    cur = src @ rot.T + trans
    d, _ = tree.query(cur)
    rms = float(np.sqrt((d[d <= np.quantile(d, args.trim)] ** 2).mean()))
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    print(f"correction: rotation {angle:.2f} deg, translation "
          f"{np.round(trans * 1000, 1)} mm (|t| {np.linalg.norm(trans) * 1000:.1f} mm), "
          f"trimmed rms {rms * 1000:.2f} mm after {k + 1} rounds")
    np.savez(args.out, rot=rot, trans=trans)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
