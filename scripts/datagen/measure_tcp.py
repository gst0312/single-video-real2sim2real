"""Measure the Robotiq 2F-85 TCP in the flange frame, from the simulated articulation.

The trajectory synthesis targets the FR3 flange with the grasp centre offset by the
gripper's actual geometry, which plan §4 says to take from the nvidia_droid USD rather
than assume. Here the environment is stepped to the open and the closed gripper state and
the two inner-finger pad bodies are read out; the grip centre is the midpoint of the pad
centres at closure, and the closing axis is the direction the pads travel. Everything is
expressed in the panda_link8 (flange) frame: link8 hangs off panda_link7 by the fixed
(0, 0, 0.107) from the kinematics, byte-identical between FR3 and Panda.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--out", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import polaris_lfhv.environments  # noqa: E402,F401
from polaris_lfhv.robot_links import collect_meshes, quat_to_mat  # noqa: E402

LINK7_TO_LINK8 = np.array([0.0, 0.0, 0.107])


def pad_state(robot, meshes):
    """World centroids of the two fingertip pad meshes, plus the flange frame."""
    names = robot.data.body_names
    pos = robot.data.body_pos_w[0].detach().cpu().numpy()
    quat = robot.data.body_quat_w[0].detach().cpu().numpy()
    pads = []
    for m in meshes:
        if "fingertips" not in m["prim"]:
            continue
        body = m["body"]
        r = quat_to_mat(quat[body])
        pads.append((m["points"] @ r.T + pos[body]).mean(0))
    assert len(pads) == 2, [m["prim"] for m in meshes if "fingertips" in m["prim"]]
    link7 = names.index("panda_link7")
    r7 = quat_to_mat(quat[link7])
    flange_pos = pos[link7] + r7 @ LINK7_TO_LINK8
    return np.stack(pads), flange_pos, r7


def main():
    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    from isaacsim.core.utils.stage import get_current_stage
    meshes = collect_meshes(get_current_stage(), robot)

    action = torch.zeros((1, 8), device=u.device)
    action[0, :7] = robot.data.default_joint_pos[0, :7]
    states = {}
    for label, grip in (("open", 0.0), ("closed", 1.0)):
        action[0, 7] = grip
        for _ in range(30):
            env.step(action, expensive=False)
        pads, flange, r7 = pad_state(robot, meshes)
        states[label] = (pads, flange, r7)
        centre = pads.mean(0)
        print(f"{label}: pad gap {np.linalg.norm(pads[0] - pads[1]):.4f} m, "
              f"grip centre in flange frame {np.round(r7.T @ (centre - flange), 4)}")

    pads_o, flange, r7 = states["open"]
    pads_c, _, _ = states["closed"]
    centre_c = r7.T @ (pads_c.mean(0) - flange)
    closing_world = (pads_c[0] - pads_o[0]) - (pads_c[1] - pads_o[1])
    closing = r7.T @ (closing_world / np.linalg.norm(closing_world))
    out = {
        "tcp_in_link8": [float(v) for v in centre_c],
        "closing_axis_in_link8": [float(v) for v in closing],
        "pad_gap_open_m": float(np.linalg.norm(pads_o[0] - pads_o[1])),
        "pad_gap_closed_m": float(np.linalg.norm(pads_c[0] - pads_c[1])),
    }
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
