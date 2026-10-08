"""Render a synthesised episode kinematically: joints and objects written in, no physics.

The point is to eyeball a phase-2 trajectory before any contact physics exists: the arm is
written to the reference joint positions and the mustard bottle to its synthesised
trajectory each step, both cameras rendered into a real-sim style video (single panel,
there is no real recording of a synthetic trajectory). Same write-in mechanics as
`compare_to_real_rollout.py`; the physics gate proper is phase 4's replay.

    scripts/polaris_env.sh python scripts/datagen/preview_episode.py \
        --episode .../traj/episodes/cond000_ep01.npz --out-dir .../traj/previews
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--episode", required=True)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--scale", type=float, default=0.5)
parser.add_argument("--out-dir", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

import r2s2r.environments  # noqa: E402,F401


def main():
    d = np.load(args_cli.episode)
    qpos = d["qpos"]
    grip = d["gripper"]
    mustard = d["mustard_traj"]        # (T, xyz + wxyz)
    cup = d["blue_cup_pose"]           # x y z qw qx qy qz
    name = Path(args_cli.episode).stem
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    grip_ids = [robot.data.joint_names.index("finger_joint")]

    cup_pose = torch.tensor([[*cup[:3], *cup[3:]]], dtype=torch.float32, device=u.device)
    u.scene["blue_cup"].write_root_pose_to_sim(cup_pose)

    writers = {}
    for cam in ("external_cam", "wrist_cam"):
        writers[cam] = imageio.get_writer(out / f"{name}_{cam}.mp4", fps=15,
                                          quality=7, macro_block_size=8)
    for t in range(0, len(qpos), args_cli.stride):
        q = torch.tensor(qpos[t], dtype=torch.float32, device=u.device)[None]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=arm_ids)
        g = torch.full((1, 1), float(grip[t]) * np.pi / 4, device=u.device)
        robot.write_joint_state_to_sim(g, torch.zeros_like(g), joint_ids=grip_ids)
        pose = torch.tensor([[mustard[t, 0], mustard[t, 1], mustard[t, 2],
                              mustard[t, 3], mustard[t, 4], mustard[t, 5], mustard[t, 6]]],
                            dtype=torch.float32, device=u.device)
        u.scene["mustard"].write_root_pose_to_sim(pose)
        u.sim.render()
        u.scene.update(0)
        rgb = u.custom_render(True, transform_static=True)
        for cam, w in writers.items():
            img = np.asarray(rgb[cam]).astype(np.uint8)
            size = (int(img.shape[1] * args_cli.scale), int(img.shape[0] * args_cli.scale))
            w.append_data(np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS)))
    for w in writers.values():
        w.close()
    print(f"wrote {out}/{name}_*.mp4 ({len(range(0, len(qpos), args_cli.stride))} frames)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
