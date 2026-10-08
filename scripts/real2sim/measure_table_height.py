"""Table height in the working base frame, measured from the real ZED depth.

Backproject the depth of static home_static frames with the intrinsics and a camera
extrinsic, keep pixels inside the object placement region, and fit a plane by trimmed
least squares. Plain pinhole backprojection; the depth is the ZED's rectified-left uint16
millimetre map.

Run as-is this reads `zed_pose_gsworld.npz`, which is the extrinsic it was run with on
2026-08-12: three frames give -0.0181 m at the placement centre (plane rms 0.7 mm each),
agreeing with the splat's own cloth fit. The wrist-camera marker solve says -0.032; that
~13 mm gap is the camera chain versus the kinematic chain.

Later the same day everything was re-anchored to the KINEMATIC base frame (the frame
joint-position actions live in) and the camera was re-solved into `zed_pose_fk.npz`; in
that frame the cloth centre sits at -0.0195, so the settled values in
`build_scene_assets.sh` are platform -0.020 and objects -0.019, not the -0.019/-0.018
this script's first run implied. Point it at `zed_pose_fk.npz` to re-measure in the
settled frame. The kinematic side gets confirmed in phase 6 by touching the real table
with the gripper.
"""
import os
import json

import numpy as np
from PIL import Image

W = os.environ.get("WORK", "work")
SEG = os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "data/random_rollouts_20260710/20260710/home_static")

pose = np.load(f"{W}/zed_pose_gsworld.npz")
cam_pos, rot = pose["pos"], pose["rot"]          # camera -> base rotation, cv convention
intr = json.load(open(os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json")))
rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]

zs = []
for idx in (100, 160, 300):
    depth = np.asarray(Image.open(f"{SEG}/cam36087771/depth/{idx:06d}.png"), np.uint16)
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth.astype(np.float64) / 1000.0
    ok = (z > 0.3) & (z < 3.0)
    pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)[ok]
    pts = pts_cam @ rot.T + cam_pos

    # object placement region, away from both mounting plates
    m = (pts[:, 0] > 0.36) & (pts[:, 0] < 0.60) & (np.abs(pts[:, 1]) < 0.24) & \
        (pts[:, 2] > -0.15) & (pts[:, 2] < 0.10)
    p = pts[m]
    for _ in range(3):
        A = np.c_[p[:, :2], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        resid = p[:, 2] - A @ coef
        p = p[np.abs(resid) < 3 * resid.std()]
    tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
    centre_z = coef[0] * 0.48 + coef[1] * 0.0 + coef[2]
    zs.append(centre_z)
    print(f"frame {idx}: {m.sum()} px, plane z@centre {centre_z:+.4f} m, "
          f"tilt {tilt:.2f} deg, rms {resid.std() * 1000:.1f} mm, kept {len(p)}")
print(f"\nmean z at placement centre: {np.mean(zs):+.4f} m")
