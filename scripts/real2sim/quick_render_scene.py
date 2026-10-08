"""Render a scene splat from the deployment camera in seconds, no Isaac.

Iterating on asset cleanups through the full environment costs ~2 minutes of Isaac boot
per look. This calls `gsplat.rasterization` directly (same backend the environment's
renderer uses, full 3-degree SH, principal point taken exactly from the intrinsics), so a
parameter sweep renders in seconds per variant.

    PATH=$POLARIS_ROOT/.venv/bin:$PATH \
    $POLARIS_ROOT/.venv/bin/python scripts/real2sim/quick_render_scene.py \
        --splat <scene.ply> --out <png>
"""

import argparse
import os
import json

import numpy as np
import torch
from plyfile import PlyData
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", nargs="+", required=True)
    p.add_argument("--pose", default=os.path.join(os.environ.get("WORK", "work"), "zed_pose_fk.npz"))
    p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", nargs="+", required=True)
    args = p.parse_args()

    from gsplat import rasterization

    intr = json.load(open(args.intrinsics))
    rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
    K = torch.tensor([[rect["fx"], 0, rect["cx"]],
                      [0, rect["fy"], rect["cy"]],
                      [0, 0, 1]], dtype=torch.float32, device="cuda")
    pose = np.load(args.pose)
    rot, pos = pose["rot"], pose["pos"]           # camera->base, cv convention
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rot.T
    view[:3, 3] = -rot.T @ pos
    viewmat = torch.from_numpy(view).cuda()

    for splat, out in zip(args.splat, args.out):
        v = PlyData.read(splat)["vertex"].data
        means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32)
        quats = torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32)
        n_scales = sum(1 for name in v.dtype.names if name.startswith("scale_"))
        s = np.stack([v[f"scale_{i}"] for i in range(n_scales)], 1)
        if n_scales == 2:
            s = np.concatenate([s, np.full((len(s), 1), -20.0)], 1)
        scales = torch.tensor(np.exp(s), dtype=torch.float32)
        opac = torch.tensor(1 / (1 + np.exp(-v["opacity"].astype(np.float64))), dtype=torch.float32)
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
        rest = np.stack([v[f"f_rest_{i}"] for i in range(45)], 1).reshape(-1, 3, 15)
        sh = np.concatenate([dc, rest.transpose(0, 2, 1)], 1)
        colors = torch.tensor(sh, dtype=torch.float32)

        img, _, _ = rasterization(
            means.cuda(), torch.nn.functional.normalize(quats.cuda(), dim=1),
            scales.cuda(), opac.cuda(), colors.cuda(),
            viewmat[None], K[None], args.width, args.height,
            sh_degree=3)
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
