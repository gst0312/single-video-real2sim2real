"""Render a splat from the deployment camera through GSWorld's own renderer.

`render_splat_from_zed_gsplat.py` goes through gsplat; this one goes through the
exact code GSWorld renders their published frames with, so a difference against
their renders can only come from the asset or the camera, not from the renderer:

- `gaussian_renderer.render` and `scene.gaussian_model.GaussianModel` are imported
  from their vendored inria 3DGS
  (`GSWorld/submodules/gaussian-splatting`, the diff_gaussian_rasterization CUDA
  rasterizer), not reimplemented. `load_ply` there sets `active_sh_degree` to the
  max, so this path is full 3rd-order SH.
- The pipeline flags are theirs (`gsworld/utils/gs_utils.py: PipelineParams`):
  convert_SHs_python / compute_cov3D_python / debug / antialiasing all False.
- Background is black and the principal point is handled by rendering a padded
  frame and cropping, both copied from `gs_world_wrapper.py`.

The camera pose is camera->base (OpenCV axes), the same thing their `right2base`
constant holds; the splat is expected to be in the robot base frame already.

Runs in the 2dgs tooling venv (torch, plyfile, simple_knn, diff_gaussian_rasterization):
    $TWODGS/.venv/bin/python scripts/real2sim/render_splat_from_zed_inria.py \
        --splat .../gs06tf_3dgs.ply --intrinsics .../zed_intrinsics_live_20260707.json \
        --pose-npz .../zed_pose_gsworld.npz --out .../gs06tf_inria.png
"""

import argparse
import os
import json
import sys

import numpy as np
from PIL import Image

GS_REPO = os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "submodules/gaussian-splatting")


class Pipe:
    """GSWorld's PipelineParams values (gs_utils.py), without the argparse plumbing."""

    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True, help="3DGS ply in the robot base frame")
    p.add_argument("--intrinsics", required=True, help="zed_intrinsics json")
    p.add_argument("--camera-key", default="36087771_left")
    p.add_argument("--pose-npz", required=True, help="npz with pos (3,) and rot (3,3), camera->base")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--centred-principal-point", action="store_true")
    p.add_argument("--gs-repo", default=GS_REPO)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, args.gs_repo)
    import torch
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    intr = json.load(open(args.intrinsics))
    entry = next(c for c in intr["cameras"] if c["serial"] == args.camera_key.split("_")[0])
    rect = entry["rectified"]["left"]
    fx, fy = rect["fx"], rect["fy"]

    dx = dy = 0
    if not args.centred_principal_point:
        dx = int(round(rect["cx"] - args.width / 2))
        dy = int(round(rect["cy"] - args.height / 2))
    render_w, render_h = args.width + 2 * abs(dx), args.height + 2 * abs(dy)
    crop_x, crop_y = abs(dx) - dx, abs(dy) - dy
    fovx = 2 * np.arctan(render_w / (2 * fx))
    fovy = 2 * np.arctan(render_h / (2 * fy))

    d = np.load(args.pose_npz)
    pos, rot = np.asarray(d["pos"], np.float64), np.asarray(d["rot"], np.float64)
    # inria's Camera takes R = camera->world rotation and T = world->camera translation
    R = rot
    T = -rot.T @ pos
    print(f"camera at {np.round(pos, 4)}, fov {np.degrees(fovx):.2f} x {np.degrees(fovy):.2f} deg, "
          f"render {render_w}x{render_h} cropped at ({crop_x}, {crop_y})")

    gaussians = GaussianModel(3)
    gaussians.load_ply(args.splat)
    print(f"{gaussians.get_xyz.shape[0]} gaussians, active sh degree {gaussians.active_sh_degree}")

    cam = Camera(resolution=(render_w, render_h), colmap_id=0, R=R, T=T,
                 FoVx=float(fovx), FoVy=float(fovy), depth_params=None,
                 image=Image.new("RGB", (args.width, args.height)), invdepthmap=None,
                 image_name="zed", uid=0, data_device="cuda")
    bg = torch.tensor(args.bg, dtype=torch.float32, device="cuda")
    out = render(cam, gaussians, Pipe(), bg, use_trained_exp=False, separate_sh=False)["render"]

    img = out.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)[crop_y:crop_y + args.height, crop_x:crop_x + args.width]
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out} {img.shape}")


if __name__ == "__main__":
    main()
