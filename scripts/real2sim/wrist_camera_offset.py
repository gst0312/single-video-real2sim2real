"""Work out where our wrist camera actually sits, in the form droid_cfg wants.

PolaRiS mounts a wrist camera on the gripper base link with a fixed offset and a fixed
field of view, both DROID's:

    pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
    focal_length=2.8, horizontal_aperture=5.376   ->  87.9 degrees across

Ours is a different camera on a different bracket: the ZED reports fx 730.59 at 1280 wide,
which is 82.4 degrees, and the 2026-07-07 calibration solved its pose against panda_link8,
not against the gripper base. Rendering the wrist view with DROID's numbers therefore shows
a view our robot never has, which is worth fixing before judging how the wrist view looks.

This composes the calibration's link8-to-camera transform with the link8-to-gripper-base
transform the USD carries, converts to the OpenGL convention IsaacLab's OffsetCfg uses
(camera looks down its own -Z, +Y up, whereas the calibration is the usual computer vision
frame looking down +Z with +Y down), and prints the CameraCfg block.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--calib", required=True)
parser.add_argument("--wrist2link-json", default=None,
                    help="json holding a 4x4 camera-in-link matrix to use instead of the "
                         "calibration's, same convention (OpenCV camera axes). GSWorld's "
                         "`wrist2eef` constant is such a matrix and is the one their splat "
                         "was built with.")
parser.add_argument("--wrist-key", default="16478870_left")
parser.add_argument("--intrinsics-serial", default="16478870")
parser.add_argument("--segment", default=None,
                    help="a recording directory, to read the camera matrix the ZED reports")
parser.add_argument("--mount", default="Gripper/Robotiq_2F_85/base_link",
                    help="prim the camera hangs off, relative to the robot root")
parser.add_argument("--link", default="panda_link8", help="body the calibration solved against")
parser.add_argument("--focal-length", type=float, default=2.8, help="kept, apertures follow it")
parser.add_argument("--report", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaacsim.core.utils.xforms import get_world_pose  # noqa: E402

import r2s2r.environments  # noqa: E402,F401
from r2s2r.robot_links import quat_to_mat  # noqa: E402


def euler_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def mat_to_quat(r):
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def main():
    report = {}
    if args_cli.wrist2link_json:
        m = np.array(json.load(open(args_cli.wrist2link_json))["matrix"], dtype=float)
        t_link_cam, r_link_cam = m[:3, 3], m[:3, :3]
        report["hand_eye_link8_to_camera"] = {"pos": t_link_cam.tolist(),
                                              "source": args_cli.wrist2link_json}
    else:
        cal = json.load(open(args_cli.calib))
        pose = cal["wrist"][args_cli.wrist_key]["pose"]
        t_link_cam, r_link_cam = np.array(pose[:3]), euler_xyz(*pose[3:6])
        report["hand_eye_link8_to_camera"] = {"pos": t_link_cam.tolist(),
                                              "euler_xyz": list(pose[3:6])}

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    u.sim.render()
    u.scene.update(0)

    link_i = robot.data.body_names.index(args_cli.link)
    r_w_link = quat_to_mat(robot.data.body_quat_w[0, link_i].detach().cpu().numpy())
    p_w_link = robot.data.body_pos_w[0, link_i].detach().cpu().numpy()

    # The mount is a plain xform under the robot, not a rigid body, so read it off the stage.
    mount_path = f"/World/envs/env_0/robot/{args_cli.mount}"
    p_w_mount, q_w_mount = get_world_pose(mount_path)
    p_w_mount = np.asarray(p_w_mount, dtype=float)
    r_w_mount = quat_to_mat(np.asarray(q_w_mount, dtype=float))

    # camera in the world, then in the mount's frame
    r_w_cam = r_w_link @ r_link_cam
    p_w_cam = p_w_link + r_w_link @ t_link_cam
    r_mount_cam = r_w_mount.T @ r_w_cam
    p_mount_cam = r_w_mount.T @ (p_w_cam - p_w_mount)

    # OffsetCfg with convention="opengl": the calibration is the computer vision frame,
    # x right, y down, z forward; OpenGL is x right, y up, z backward.
    r_gl = r_mount_cam @ np.diag([1.0, -1.0, -1.0])
    quat = mat_to_quat(r_gl)

    report["offset"] = {"pos": [float(v) for v in p_mount_cam],
                        "rot_opengl_wxyz": [float(v) for v in quat],
                        "mount": args_cli.mount}

    if args_cli.segment:
        meta = json.load(open(Path(args_cli.segment)
                              / f"cam{args_cli.intrinsics_serial}" / "meta.json"))
        k = np.array(meta["K"], dtype=float)
        w, h = meta["resolution"]
        f = args_cli.focal_length
        report["intrinsics"] = {
            "width": w, "height": h,
            "focal_length": f,
            "horizontal_aperture": float(2 * f * (w / 2) / k[0, 0]),
            "vertical_aperture": float(2 * f * (h / 2) / k[1, 1]),
            "fov_x_deg": float(np.degrees(2 * np.arctan((w / 2) / k[0, 0]))),
            "fov_y_deg": float(np.degrees(2 * np.arctan((h / 2) / k[1, 1]))),
        }

    with open(args_cli.report, "w") as f:
        json.dump(report, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
