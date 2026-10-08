"""Refine the wrist camera mount against the solved marker positions.

The wrist view draws the table a few pixels off while the external view is aligned, which
points at the one transform only the wrist depends on: the camera-in-link mount. Both
candidates on file are second-hand for the current scene anchor - GSWorld's `wrist2eef`
was right for the splat THEIR mount built, and the 2026-07-07 hand-eye was solved before
the scene was re-anchored to the kinematic frame - so this solves the mount against that
anchor directly.

Same data flow as `markers_in_base_from_wrist.py`, inverted: there the mount was trusted
and the markers were solved; here the markers (solved from the external camera chain and
recorded in `markers_in_base.json`) are trusted and the mount is least-squares fitted to
the wrist detections. Each detection contributes the reprojection of the four known
marker corners through FK and the candidate mount; the report compares both stored
candidates before refinement and prints the refined mount as the `CameraCfg.OffsetCfg`
block `droid_cfg` wants (conversion identical to `wrist_camera_offset.py`).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--segments", nargs="+", required=True)
parser.add_argument("--calib", required=True)
parser.add_argument("--markers", required=True, help="markers_in_base.json")
parser.add_argument("--wrist2link-json", default=None,
                    help="json with a 4x4 camera-in-link matrix used as the second "
                         "candidate start (GSWorld's wrist2eef)")
parser.add_argument("--wrist-key", default="16478870_left")
parser.add_argument("--serial", default="16478870")
parser.add_argument("--stride", type=int, default=6)
parser.add_argument("--marker-size", type=float, default=0.1)
parser.add_argument("--link", default="panda_link8")
parser.add_argument("--mount", default="Gripper/Robotiq_2F_85/base_link")
parser.add_argument("--out", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaacsim.core.utils.xforms import get_world_pose  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import r2s2r.environments  # noqa: E402,F401


def euler_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def mat_to_quat(r):
    q = Rotation.from_matrix(r).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cal = json.load(open(args_cli.calib))
    pose = cal["wrist"][args_cli.wrist_key]["pose"]
    starts = {"calib_2026_07": (np.array(pose[:3]), euler_xyz(*pose[3:6]))}
    if args_cli.wrist2link_json:
        m = np.array(json.load(open(args_cli.wrist2link_json))["matrix"], dtype=float)
        starts["wrist2eef"] = (m[:3, 3].copy(), m[:3, :3].copy())

    known = json.load(open(args_cli.markers))
    mk_pos = np.array([m["pos"] for m in known["markers"]])
    mk_rot = np.array([m["rot"] for m in known["markers"]])
    half = args_cli.marker_size / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    mk_corners = mk_pos[:, None, :] + np.einsum("mij,cj->mci", mk_rot, obj)

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    link = robot.data.body_names.index(args_cli.link)
    base = robot.data.body_names.index("panda_link0")

    # Capture the fixed link8 -> mount transform NOW, while both pose sources agree:
    # body poses come from physics and follow every joint write, get_world_pose reads
    # USD which stays at the reset pose, so after the detection loop the two describe
    # different configurations and composing them produces garbage. It is not a pure
    # translation either - the gripper base frame is rotated relative to link8.
    u.sim.render()
    u.scene.update(0)
    r_l8 = quat_to_mat(robot.data.body_quat_w[0, link].detach().cpu().numpy())
    p_l8 = robot.data.body_pos_w[0, link].detach().cpu().numpy()
    p_m, q_m = get_world_pose(f"/World/envs/env_0/robot/{args_cli.mount}")
    r_m = quat_to_mat(np.asarray(q_m, dtype=float))
    r_l8_mount = r_l8.T @ r_m
    t_l8_mount = r_l8.T @ (np.asarray(p_m, dtype=float) - p_l8)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    # collect (K, corners2d, T_base_link) per detection; marker matched later per mount
    obs = []
    for seg_path in args_cli.segments:
        seg = Path(seg_path)
        cam_dir = seg / f"cam{args_cli.serial}"
        meta = json.load(open(cam_dir / "meta.json"))
        K = np.array(meta["K"], dtype=np.float64)
        states = read_jsonl(seg / "robot_state.jsonl")
        state_t = np.array([s["t"] for s in states])
        state_q = np.array([s["qpos"] for s in states])
        found = 0
        for rec in read_jsonl(cam_dir / "timestamps.jsonl")[:: args_cli.stride]:
            path = cam_dir / "rgb" / f"{int(rec['i']):06d}.png"
            if not path.exists():
                continue
            gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or not len(ids):
                continue
            k = int(np.argmin(np.abs(state_t - rec["t"])))
            if abs(state_t[k] - rec["t"]) > 0.05:
                continue
            pos = torch.tensor(state_q[k], dtype=torch.float32, device=u.device)[None]
            robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=arm_ids)
            u.sim.render()
            u.scene.update(0)
            r_wb = quat_to_mat(robot.data.body_quat_w[0, base].detach().cpu().numpy())
            p_b = robot.data.body_pos_w[0, base].detach().cpu().numpy()
            p_l = robot.data.body_pos_w[0, link].detach().cpu().numpy()
            r_l = quat_to_mat(robot.data.body_quat_w[0, link].detach().cpu().numpy())
            t_b_link = r_wb.T @ (p_l - p_b)
            r_b_link = r_wb.T @ r_l
            for c in corners:
                obs.append({"seg": seg.name, "K": K, "px": c.reshape(4, 2).astype(np.float64),
                            "t_b_link": t_b_link, "r_b_link": r_b_link})
                found += 1
        print(f"{seg.name}: {found} detections", flush=True)
    print(f"{len(obs)} detections in total")
    if len(obs) < 30:
        raise SystemExit("too few detections to fit a mount")

    def residuals(t_link_cam, r_link_cam, match=None):
        res, used = [], []
        for i, o in enumerate(obs):
            r_b_cam = o["r_b_link"] @ r_link_cam
            t_b_cam = o["t_b_link"] + o["r_b_link"] @ t_link_cam
            best = None
            candidates = range(len(mk_pos)) if match is None else [match[i]]
            for m in candidates:
                pts = (mk_corners[m] - t_b_cam) @ r_b_cam  # rows: corner in cam frame
                if pts[:, 2].min() < 0.05:
                    continue
                uv = pts[:, :2] / pts[:, 2:3] * [o["K"][0, 0], o["K"][1, 1]] \
                    + [o["K"][0, 2], o["K"][1, 2]]
                err = uv - o["px"]
                if best is None or np.abs(err).mean() < np.abs(best[0]).mean():
                    best = (err, m)
            if best is None:
                continue
            res.append(best[0].ravel())
            used.append((i, best[1]))
        return np.concatenate(res), used

    report = {"n_detections": len(obs), "starts": {}}
    best_start, best_rms = None, np.inf
    for name, (t0, r0) in starts.items():
        r, _ = residuals(t0, r0)
        rms = float(np.sqrt((r ** 2).mean()))
        report["starts"][name] = {"rms_px": rms, "pos": t0.tolist()}
        print(f"start {name}: reprojection rms {rms:.2f} px over {len(r) // 8} detections")
        if rms < best_rms:
            best_start, best_rms = name, rms
    t0, r0 = starts[best_start]

    # freeze the detection->marker matching at the better start, then refine 6 dof
    _, used = residuals(t0, r0)
    match = {i: m for i, m in used}
    obs[:] = [obs[i] for i in match]
    match = {j: match[i] for j, i in enumerate(sorted(match))}

    def pack_residual(x):
        dr = Rotation.from_rotvec(x[:3]).as_matrix()
        r, _ = residuals(t0 + x[3:], dr @ r0, match)
        return r

    fit = least_squares(pack_residual, np.zeros(6), loss="soft_l1", f_scale=3.0)
    r_ref = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ r0
    t_ref = t0 + fit.x[3:]
    r, _ = residuals(t_ref, r_ref, match)
    rms = float(np.sqrt((r ** 2).mean()))
    dt = np.linalg.norm(fit.x[3:]) * 1000
    da = np.degrees(np.linalg.norm(fit.x[:3]))
    print(f"refined from {best_start}: rms {best_rms:.2f} -> {rms:.2f} px "
          f"(moved {dt:.1f} mm, {da:.2f} deg)")
    report["refined"] = {"from": best_start, "rms_px": rms,
                         "moved_mm": dt, "rotated_deg": da,
                         "link8_to_camera": {"pos": t_ref.tolist(), "rot": r_ref.tolist()}}

    # per-segment rms with the refined mount, to spot a bad recording
    seg_err = {}
    for o, e in zip([obs[i] for i in match], np.split(r, len(r) // 8)):
        seg_err.setdefault(o["seg"], []).append(e)
    report["per_segment_rms_px"] = {
        s: float(np.sqrt((np.concatenate(v) ** 2).mean())) for s, v in seg_err.items()}
    for s, v in report["per_segment_rms_px"].items():
        print(f"  {s}: {v:.2f} px")

    # convert to the OffsetCfg block through the fixed transform captured at reset
    # (validated: run on the raw 2026-07 calibration this reproduces the OffsetCfg in
    # PourMustardCfg's docstring to six decimals)
    r_mount_cam = r_l8_mount.T @ r_ref
    p_mount_cam = r_l8_mount.T @ (t_ref - t_l8_mount)
    quat = mat_to_quat(r_mount_cam @ np.diag([1.0, -1.0, -1.0]))
    report["offset"] = {"pos": [float(v) for v in p_mount_cam],
                        "rot_opengl_wxyz": [float(v) for v in quat],
                        "mount": args_cli.mount}
    print("OffsetCfg pos", np.round(p_mount_cam, 6).tolist(),
          "rot", np.round(quat, 6).tolist())

    Path(args_cli.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
