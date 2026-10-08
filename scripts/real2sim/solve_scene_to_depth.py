"""Anchor the converted scene splat to the kinematic base frame, against real depth.

With the camera solved off the FK arm (`solve_camera_from_depth.py`), backprojected depth
of the static scene lives in the kinematic frame with no further assumptions. This solves
the remaining rigid transform of the scene splat against that cloud - replacing the
marker-frame detour: the two-marker alignment anchors roll and pitch to the marker plane,
i.e. to the tabletop, and the physical tabletop is itself tilted ~1.3 deg against the
robot base plane, so "marker frame" and "kinematic frame" genuinely differ.

  source: visible background gaussians of the converted scene (label -1), tabletop and
          mounting-plate band only (z < 0.25 in its own frame, inside the table footprint);
  target: depth pixels of static frames, arm masked away by the FK gate, backprojected
          with the FK-solved camera;
  solver: trimmed nearest neighbour + rigid Umeyama, initialised at identity (the scene
          already carries the config transform and the marker-frame correction).

The result composes onto the existing extra transform and feeds the build as the new
`--extra-transform`.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2s2r.alignment import umeyama  # noqa: E402

ROLLOUTS = Path(os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "data/random_rollouts_20260710/20260710"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="converted scene splat, robot split away")
    p.add_argument("--camera", default=os.path.join(os.environ.get("WORK", "work"), "zed_pose_fk.npz"))
    p.add_argument("--prev-transform",
                   default=os.path.join(os.environ.get("WORK", "work"), "frame_correction.npz"),
                   help="the transform the scene already carries; composed into the output")
    p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
    p.add_argument("--frames", nargs="*", default=["home_static:100", "home_static:300"])
    p.add_argument("--trim", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    intr = json.load(open(args.intrinsics))
    rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
    fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]
    cam = np.load(args.camera)
    c_rot, c_pos = cam["rot"].astype(np.float64), cam["pos"].astype(np.float64)

    from PIL import Image
    tgt = []
    for spec in args.frames:
        seg, idx = spec.split(":")
        idx = int(idx)
        rgb = np.asarray(Image.open(ROLLOUTS / seg / f"cam36087771/rgb/{idx:06d}.png").convert("RGB"))
        depth = np.asarray(Image.open(ROLLOUTS / seg / f"cam36087771/depth/{idx:06d}.png"), np.uint16)
        lum = rgb.astype(np.float64).mean(2)
        ok = (depth > 300) & (depth < 2200) & (lum < 140)   # static and not the white arm
        v, u = np.nonzero(ok)
        z = depth[v, u] / 1000.0
        pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1) @ c_rot.T + c_pos
        # tabletop and plate band only
        keep = (pts[:, 2] > -0.15) & (pts[:, 2] < 0.25) & \
               (pts[:, 0] > -0.3) & (pts[:, 0] < 0.9) & (np.abs(pts[:, 1]) < 0.7)
        tgt.append(pts[keep])
        print(f"{seg} f{idx}: {int(keep.sum())} static points")
    tgt = np.concatenate(tgt)

    vert = PlyData.read(args.scene)["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    opa = vert["opacity"].astype(np.float64)
    src = xyz[(opa > -3) & (xyz[:, 2] > -0.15) & (xyz[:, 2] < 0.25) &
              (xyz[:, 0] > -0.3) & (xyz[:, 0] < 0.9) & (np.abs(xyz[:, 1]) < 0.7)]
    rng = np.random.default_rng(0)
    if len(src) > 40000:
        src = src[rng.choice(len(src), 40000, replace=False)]
    print(f"{len(src)} scene gaussians vs {len(tgt)} depth points")

    tree = cKDTree(tgt)
    rot, trans = np.eye(3), np.zeros(3)
    for k in range(args.rounds):
        cur = src @ rot.T + trans
        d, idx_ = tree.query(cur)
        keep = d <= np.quantile(d, args.trim)
        _, rot_n, trans_n = umeyama(src[keep], tgt[idx_[keep]], with_scale=False)
        if np.allclose(rot_n, rot, atol=1e-10) and np.allclose(trans_n, trans, atol=1e-10):
            break
        rot, trans = rot_n, trans_n
    cur = src @ rot.T + trans
    d, _ = tree.query(cur)
    rms = float(np.sqrt((d[d <= np.quantile(d, args.trim)] ** 2).mean()))
    ang = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    print(f"scene refine: rotation {ang:.2f} deg, translation {np.round(trans * 1000, 1)} mm, "
          f"trimmed rms {rms * 1000:.2f} mm, {k + 1} rounds")

    prev = np.load(args.prev_transform)
    rot_out = rot @ prev["rot"]
    trans_out = rot @ prev["trans"] + trans
    np.savez(args.out, rot=rot_out, trans=trans_out)
    print(f"wrote {args.out} (composed onto {args.prev_transform})")


if __name__ == "__main__":
    main()
