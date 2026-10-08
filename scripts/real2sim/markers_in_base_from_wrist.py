"""Locate both table markers in the robot base frame, from the wrist camera recordings.

The 2026-07-07 calibration solved one marker this way and stopped there: its
`marker2base` has a single entry. GSWorld's notes describe the recipe as averaging
`T_g2b · T_c2g · T_m2c` over frames, 57 of them, to 2.1 mm and 0.21 degrees. The same
recipe applies to the second marker, and there is a second marker sitting next to the
robot base — the recordings just have to be read.

Why it matters: the scene alignment is currently anchored on four corners of one 100 mm
square. Two markers 0.41 m apart give a baseline four times longer, which is exactly the
direction the current fit is weak in.

Forward kinematics comes from the articulation PolaRiS evaluates with; the hand-eye result
is the calibration's own; marker pose in the camera is OpenCV's IPPE_SQUARE solver.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--segments", nargs="+", required=True)
parser.add_argument("--calib", required=True)
parser.add_argument("--wrist2link-json", default=None,
                    help="json with a 4x4 camera-in-link matrix to use instead of the "
                         "calibration's hand-eye, same convention. The marker heights this "
                         "script solves are only as good as the hand-eye, so re-running with "
                         "GSWorld's `wrist2eef` is the independent check on our table height.")
parser.add_argument("--wrist-key", default="16478870_left")
parser.add_argument("--serial", default="16478870")
parser.add_argument("--stride", type=int, default=6, help="use every Nth frame of each segment")
parser.add_argument("--marker-size", type=float, default=0.1, help="black square side, metres")
parser.add_argument("--link", default="panda_link8")
parser.add_argument("--cluster-radius", type=float, default=0.12)
parser.add_argument("--out", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

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


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cal = json.load(open(args_cli.calib))
    if args_cli.wrist2link_json:
        m = np.array(json.load(open(args_cli.wrist2link_json))["matrix"], dtype=float)
        t_link_cam, r_link_cam = m[:3, 3], m[:3, :3]
        print(f"hand-eye from {args_cli.wrist2link_json}")
    else:
        pose = cal["wrist"][args_cli.wrist_key]["pose"]
        t_link_cam = np.array(pose[:3])
        r_link_cam = euler_xyz(*pose[3:6])

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    link = robot.data.body_names.index(args_cli.link)
    base = robot.data.body_names.index("panda_link0")

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    half = args_cli.marker_size / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])

    detections = []
    for seg_path in args_cli.segments:
        seg = Path(seg_path)
        cam_dir = seg / f"cam{args_cli.serial}"
        meta = json.load(open(cam_dir / "meta.json"))
        K = np.array(meta["K"], dtype=np.float64)
        states = read_jsonl(seg / "robot_state.jsonl")
        state_t = np.array([s["t"] for s in states])
        state_q = np.array([s["qpos"] for s in states])
        stamps = read_jsonl(cam_dir / "timestamps.jsonl")[:: args_cli.stride]
        found = 0
        for rec in stamps:
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
            r_b_cam = r_b_link @ r_link_cam
            t_b_cam = t_b_link + r_b_link @ t_link_cam

            for c in corners:
                ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, None,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                r_cam_mk, _ = cv2.Rodrigues(rvec)
                t_b_mk = t_b_cam + r_b_cam @ tvec.ravel()
                r_b_mk = r_b_cam @ r_cam_mk
                detections.append({"seg": seg.name, "frame": int(rec["i"]),
                                   "pos": t_b_mk.tolist(), "rot": r_b_mk.tolist(),
                                   "dist": float(np.linalg.norm(tvec))})
                found += 1
        print(f"{seg.name}: {found} marker detections")

    pos = np.array([d["pos"] for d in detections])
    print(f"\n{len(detections)} detections in total")
    if not len(pos):
        raise SystemExit("nothing detected")

    # cluster by position; the two markers are far apart compared with the noise
    labels = -np.ones(len(pos), int)
    next_label = 0
    for i in range(len(pos)):
        if labels[i] >= 0:
            continue
        near = np.linalg.norm(pos - pos[i], axis=1) < args_cli.cluster_radius
        labels[near & (labels < 0)] = next_label
        next_label += 1
    print(f"clusters: {next_label}")

    out = {"marker_size": args_cli.marker_size, "markers": []}
    known = np.array(cal["marker2base"]["4x4:n0"]["pose"][:3])
    for lab in range(next_label):
        sel = labels == lab
        if sel.sum() < 5:
            continue
        p = pos[sel]
        rots = np.array([detections[i]["rot"] for i in np.where(sel)[0]])
        um, _, vt = np.linalg.svd(rots.mean(axis=0))
        rot = um @ vt
        normal = rot[:, 2]
        if normal[2] < 0:
            normal = -normal
        entry = {"n": int(sel.sum()), "pos": p.mean(axis=0).tolist(),
                 "std_mm": (p.std(axis=0) * 1000).tolist(), "rot": rot.tolist(),
                 "normal": normal.tolist()}
        out["markers"].append(entry)
        print(f"cluster {lab}: {int(sel.sum())} detections")
        print(f"  centre {np.round(p.mean(axis=0), 4)}  std {np.round(p.std(axis=0) * 1000, 1)} mm")
        print(f"  plane normal {np.round(normal, 4)}")
        print(f"  distance to the calibration's marker: "
              f"{np.linalg.norm(p.mean(axis=0) - known) * 1000:.0f} mm")

    if len(out["markers"]) >= 2:
        a = np.array(out["markers"][0]["pos"])
        b = np.array(out["markers"][1]["pos"])
        print(f"\nseparation between the two markers: {np.linalg.norm(a - b) * 1000:.1f} mm")
    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
