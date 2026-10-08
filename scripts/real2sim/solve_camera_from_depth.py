"""Solve the deployment camera's pose in the robot base frame from real depth + FK.

The camera extrinsic inherited from GSWorld leaves image residuals that grow towards the
top of the frame (10 px at the arm's top, 0 at the near table edge); co-transforming
scene and camera cannot change them, because they measure camera-versus-world. This
anchors the camera to the one thing whose world geometry is known exactly - the robot,
via forward kinematics of the recorded joint angles - using the ZED's own depth:

  per frame: arm pixels (the white arm against the black backdrop, luminance threshold
  inside a generous region prior) are backprojected with the rectified intrinsics into
  camera-frame 3D points; the FR3 URDF meshes, posed by FK at that frame's qpos, are
  point-sampled in the base frame; a trimmed ICP over all frames jointly solves the rigid
  camera-to-base transform. The gripper is not in the FR3 URDF, so points near and below
  the flange are dropped from both clouds.

With the camera solved, the tabletop pixels backproject into the base frame with no
further assumptions, giving the table height on the kinematic chain directly.

Runs in the traj venv (jaxmp FK + yourdfpy meshes + scipy):
    JAX_PLATFORMS=cpu $TRAJ_VENV/bin/python \
        scripts/real2sim/solve_camera_from_depth.py --out $WORK/zed_pose_fk.npz
"""

import argparse
import os
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polaris_lfhv.alignment import umeyama  # noqa: E402

ROLLOUTS = Path(os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "data/random_rollouts_20260710/20260710"))
FRAMES = [("home_static", 160), ("randomwalk_1", 400), ("randomwalk_2", 660),
          ("randomwalk_3", 300), ("randomwalk_4", 500), ("vertical_1", 110),
          ("wristroll_1", 400), ("randomwalk_2", 200)]


def fk_arm_points(urdf, kin, qpos, per_link=1500, seed=0):
    """Surface samples of the posed arm in the base frame, links 1..7 (no hand)."""
    cfg = np.concatenate([qpos, np.zeros(kin.num_actuated_joints - 7)])
    tf = np.asarray(jnp.reshape(kin.forward_kinematics(jnp.asarray(cfg)[None]), (-1, 7)))
    rng = np.random.default_rng(seed)
    pts = []
    # yourdfpy: walk links, find the joint index driving each link. link0 is fixed to the
    # base and has no driving joint in the kinematic tree: identity frame.
    for link_name, link in urdf.link_map.items():
        if not link.visuals or "finger" in link_name or "hand" in link_name:
            continue
        parent_joint = next((j for j in urdf.robot.joints if j.child == link_name), None)
        if link_name.endswith("link0"):
            rot = np.eye(3)
            tx = ty = tz = 0.0
        elif parent_joint is None or parent_joint.name not in kin.joint_names:
            continue
        else:
            ji = kin.joint_names.index(parent_joint.name)
            w, x, y, z, tx, ty, tz = tf[ji]
            rot = Rotation.from_quat([x, y, z, w]).as_matrix()
        for vis in link.visuals:
            geom = vis.geometry
            if geom.mesh is None:
                continue
            fname = urdf._filename_handler(geom.mesh.filename)
            mesh = trimesh.load(fname, force="mesh")
            if geom.mesh.scale is not None:
                mesh.apply_scale(geom.mesh.scale)
            if vis.origin is not None:
                mesh.apply_transform(vis.origin)
            s, _ = trimesh.sample.sample_surface(mesh, per_link, seed=int(rng.integers(1 << 31)))
            pts.append(s @ rot.T + np.array([tx, ty, tz]))
    return np.concatenate(pts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--intrinsics", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
    p.add_argument("--init-pose", default=os.path.join(os.environ.get("WORK", "work"), "zed_pose_gsworld.npz"))
    p.add_argument("--lum-min", type=float, default=120.0, help="arm pixels are bright")
    p.add_argument("--region", type=int, nargs=4, default=[0, 480, 380, 1000],
                   help="y0 y1 x0 x1 prior for our arm in the image")
    p.add_argument("--trim", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    intr = json.load(open(args.intrinsics))
    rect = next(c for c in intr["cameras"] if c["serial"] == "36087771")["rectified"]["left"]
    fx, fy, cx, cy = rect["fx"], rect["fy"], rect["cx"], rect["cy"]

    urdf = load_urdf("fr3_description")
    kin = JaxKinTree.from_urdf(urdf)

    cam_pts_all, fk_pts_all = [], []
    y0, y1, x0, x1 = args.region
    for seg, idx in FRAMES:
        segd = ROLLOUTS / seg
        states = [json.loads(l) for l in open(segd / "robot_state.jsonl") if l.strip()]
        stamps = {int(r["i"]): r["t"] for r in
                  (json.loads(l) for l in open(segd / "cam36087771/timestamps.jsonl"))}
        t = stamps[idx]
        ts = np.array([s["t"] for s in states])
        q = np.array(states[int(np.argmin(np.abs(ts - t)))]["qpos"])[:7]

        rgb = np.asarray(Image.open(segd / f"cam36087771/rgb/{idx:06d}.png").convert("RGB"))
        depth = np.asarray(Image.open(segd / f"cam36087771/depth/{idx:06d}.png"), np.uint16)
        lum = rgb.astype(np.float64).mean(2)
        m = np.zeros_like(lum, bool)
        m[y0:y1, x0:x1] = True
        m &= (lum > args.lum_min) & (depth > 300) & (depth < 2500)
        v, u = np.nonzero(m)
        z = depth[v, u] / 1000.0
        cam_pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
        fk_pts = fk_arm_points(urdf, kin, q)
        # gate with the initial pose: bright pixels further than 0.12 m from the posed arm
        # are the mounting plates, the neighbour arm and the backdrop, not our arm. The
        # bottom slice goes on both sides: the mounting plate hugs link0 and is bright,
        # and it is not part of the URDF.
        fk_pts = fk_pts[fk_pts[:, 2] > 0.03]
        init0 = np.load(args.init_pose)
        gated = cam_pts @ init0["rot"].T + init0["pos"]
        d0, _ = cKDTree(fk_pts).query(gated)
        cam_pts = cam_pts[(d0 < 0.12) & (gated[:, 2] > 0.03)]
        cam_pts_all.append(cam_pts)
        fk_pts_all.append(fk_pts)
        print(f"{seg} f{idx}: {len(cam_pts)} arm pixels after gating, {len(fk_pts)} fk points")

    # correspondences must stay within their own frame: each frame's pixels can only
    # match that frame's arm pose
    rng = np.random.default_rng(0)
    per_frame = 8000
    cam_pts_all = [c[rng.choice(len(c), min(per_frame, len(c)), replace=False)]
                   for c in cam_pts_all]
    trees = [cKDTree(f) for f in fk_pts_all]

    init = np.load(args.init_pose)
    rot, trans = init["rot"].astype(np.float64), init["pos"].astype(np.float64)
    for k in range(args.rounds):
        src, dst, dists = [], [], []
        for c, f, tree in zip(cam_pts_all, fk_pts_all, trees):
            cur = c @ rot.T + trans
            d, idx_ = tree.query(cur)
            src.append(c)
            dst.append(f[idx_])
            dists.append(d)
        src = np.concatenate(src)
        dst = np.concatenate(dst)
        dists = np.concatenate(dists)
        keep = dists <= np.quantile(dists, args.trim)
        _, rot_n, trans_n = umeyama(src[keep], dst[keep], with_scale=False)
        if np.allclose(rot_n, rot, atol=1e-10) and np.allclose(trans_n, trans, atol=1e-10):
            break
        rot, trans = rot_n, trans_n
    rms = float(np.sqrt((dists[keep] ** 2).mean()))
    d_init = np.linalg.norm(trans - init["pos"])
    ang = np.degrees(np.arccos(np.clip((np.trace(init["rot"].T @ rot) - 1) / 2, -1, 1)))
    print(f"camera: moved {d_init * 1000:.1f} mm, rotated {ang:.2f} deg from init; "
          f"trimmed rms {rms * 1000:.2f} mm over {int(keep.sum())} pts, {k + 1} rounds")
    print(f"pos {np.round(trans, 4)}")
    np.savez(args.out, pos=trans, rot=rot)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
