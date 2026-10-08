"""render_splat_from_zed.py, but through the gsplat-3dgs-renderer backend.

Same camera plumbing as `render_splat_from_zed.py`; the only difference is the
renderer class: `polaris.splat_renderer.gsplat_renderer.SplatRenderer` from the
`gsplat-3dgs-renderer` branch of github.com/zubair-irshad/polaris (1e3a6d4, local
checkout under refs/). That class mirrors the official interface (init_cameras /
render_raw / set_extrinsics take the same arguments, verified against both
sources), rasterizes true 3DGS via `gsplat.rasterization`, and pads 2DGS plys
with a tiny third axis. Our working copy of the branch also evaluates full
3rd-order SH.

The branch's `src` goes first on sys.path so its `polaris` package shadows the
official one for this process only.

Two conventions differ from the stock PolaRiS renderer, both taken from what the
splats were actually trained with and from how GSWorld renders the same file:

- Background. PolaRiS composites over 0.5 grey (`splat_renderer.py:16`), but
  every splat here - GSWorld's 3DGS and our own 2DGS - was trained with
  `_white_background = False`, i.e. black. Grey shows through wherever alpha < 1,
  which on the semi-transparent curtain is a heavy veil: it made our render 1.7x
  brighter than GSWorld's own render of the same gaussians. Default is black.

- Principal point. The rasterizer assumes a centred principal point, but the
  rectified ZED K is off centre (cx 638.02, cy 357.27 at 1280x720), so a centred
  render lands 2-3 px off the real image. GSWorld handles this by rendering a
  symmetrically padded frame and cropping the window that matches (cx, cy)
  (`gs_world_wrapper.py: cam_maniskill2gs`); the same arithmetic is copied here.
"""

import argparse
import os
import json
import sys

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
_args_cli, _ = _parser.parse_known_args()
_args_cli.headless = True
_app = AppLauncher(_args_cli).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def euler_xyz_to_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--camera-key", default="36087771_left")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pose-npz", default=None)
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="background the render composites over; black is what the "
                        "splats were trained with, PolaRiS's own default is 0.5 grey")
    p.add_argument("--centred-principal-point", action="store_true",
                   help="ignore cx, cy and render centred, as the stock scripts did")
    p.add_argument("--polaris-3dgs", default=os.environ.get("POLARIS_3DGS", "polaris-3dgs"),
                   help="checkout of the gsplat-3dgs-renderer branch")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.polaris_3dgs) + "/src")
    from polaris.splat_renderer.gsplat_renderer import SplatRenderer  # noqa: E402
    import polaris  # noqa: E402
    print(f"polaris package resolved to {polaris.__file__}")

    calib = json.load(open(args.calib))
    pose = calib["third"][args.camera_key]["pose"]
    cam_pos = np.array(pose[0:3])
    rot_cam_to_base = euler_xyz_to_matrix(*pose[3:6])
    if args.pose_npz:
        d = np.load(args.pose_npz)
        cam_pos, rot_cam_to_base = d["pos"], d["rot"]
        print(f"overriding the pose from {args.pose_npz}")

    intr = json.load(open(args.intrinsics))
    entry = next(c for c in intr["cameras"] if c["serial"] == args.camera_key.split("_")[0])
    rect = entry["rectified"]["left"]
    fx, fy = rect["fx"], rect["fy"]

    # off-centre principal point -> render padded, crop the matching window
    dx = dy = 0
    if not args.centred_principal_point:
        dx = int(round(rect["cx"] - args.width / 2))
        dy = int(round(rect["cy"] - args.height / 2))
    render_w, render_h = args.width + 2 * abs(dx), args.height + 2 * abs(dy)
    crop_x, crop_y = abs(dx) - dx, abs(dy) - dy
    fovx = 2 * np.arctan(render_w / (2 * fx))
    fovy = 2 * np.arctan(render_h / (2 * fy))
    print(f"camera at {np.round(cam_pos, 4)}, fov {np.degrees(fovx):.2f} x {np.degrees(fovy):.2f} deg, "
          f"render {render_w}x{render_h} cropped at ({crop_x}, {crop_y})")

    renderer = SplatRenderer(splats={"scene": args.splat}, bg_color=tuple(args.bg), device=0)
    renderer.init_cameras({"zed": {"res": (render_h, render_w), "fovx": fovx, "fovy": fovy}})
    images = renderer.render_raw({"zed": {"pos": cam_pos, "rot": rot_cam_to_base}})

    img = images["zed"].detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = img[crop_y:crop_y + args.height, crop_x:crop_x + args.width]
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out} {img.shape}")


if __name__ == "__main__":
    main()
    _app.close()
