"""Compare the environment against the human demonstration frame, objects included.

The recorded robot rollouts have an empty table, so they say nothing about how the two
task objects sit in the composite. The demonstration recording does have them, was shot
with the same ZED from the same place (the two markers land within a couple of pixels of
where the rollout frames put them), and its first frames are before the hand enters.

Object placement is solved from the picture rather than taken from a tracker: the camera
pose and the tabletop plane are both known, so the ray through a clicked base-centre pixel
meets the table at one point. That keeps this comparison free of the FoundationPose
products, which are marked for replacement.

    scripts/polaris_env.sh python scripts/real2sim/compare_to_demo_frame.py \
        --frame .../mustard_pour_take4_trim247-530/rgb/000000.png \
        --mustard-px 618 543 --cup-px 785 548 --out-dir .../demo_cmp
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--frame", required=True, help="a demonstration RGB frame")
parser.add_argument("--pose-npz", default=os.path.join(os.environ.get("WORK", "work"), "zed_pose_gsworld.npz"))
parser.add_argument("--intrinsics",
                    default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "real_robot_data/cameras/zed_intrinsics_live_20260707.json"))
parser.add_argument("--camera-key", default="36087771_left")
parser.add_argument("--table-z", type=float, default=-0.019)
parser.add_argument("--mustard-px", type=float, nargs=2, required=True,
                    help="pixel at the centre of the bottle's base")
parser.add_argument("--cup-px", type=float, nargs=2, required=True)
parser.add_argument("--mustard-yaw", type=float, default=90.0,
                    help="degrees about z. The label sits on the mesh's +Z side, which the "
                         "upright rotation turns to world -Y; 90 deg faces it at the "
                         "deployment camera on +X, matching how the bottle is put down in "
                         "the demonstrations (label towards the camera)")
parser.add_argument("--cup-yaw", type=float, default=0.0)
parser.add_argument("--qpos-segment", default=os.path.join(os.environ.get("GSWORLD_ROOT", "GSWorld"), "data/random_rollouts_20260710/20260710/home_static"),
                    help="the arm is at its home pose in the demonstration; take the angles "
                         "from the recorded home segment rather than guessing")
parser.add_argument("--out-dir", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

import r2s2r.environments  # noqa: E402,F401


def ray_to_table(px, pos, rot, fx, fy, cx, cy, z):
    """Where the ray through a pixel meets the plane z = table_z, in the base frame."""
    d_cam = np.array([(px[0] - cx) / fx, (px[1] - cy) / fy, 1.0])
    d = rot @ d_cam
    return pos + d * ((z - pos[2]) / d[2])


def main():
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(args_cli.pose_npz)
    pos, rot = np.asarray(d["pos"], float), np.asarray(d["rot"], float)
    intr = json.load(open(args_cli.intrinsics))
    rect = next(c for c in intr["cameras"]
                if c["serial"] == args_cli.camera_key.split("_")[0])["rectified"]["left"]

    places = {}
    for name, px, yaw in (("mustard", args_cli.mustard_px, args_cli.mustard_yaw),
                          ("blue_cup", args_cli.cup_px, args_cli.cup_yaw)):
        p = ray_to_table(px, pos, rot, rect["fx"], rect["fy"], rect["cx"], rect["cy"],
                         args_cli.table_z)
        half = np.radians(yaw) / 2
        # the asset frames lie on their side, the same convention initial_conditions.json uses
        q = np.array([np.cos(half), np.cos(half), np.sin(half), np.sin(half)]) / np.sqrt(2)
        places[name] = np.concatenate([p, q])
        print(f"{name}: pixel {px} -> base {np.round(p, 4)}")

    states = [json.loads(line) for line in open(Path(args_cli.qpos_segment) / "robot_state.jsonl")
              if line.strip()]
    qpos = np.array(states[len(states) // 2]["qpos"])[:7]
    print(f"arm at the recorded home pose {np.round(qpos, 4)}")

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    t = torch.tensor(qpos, dtype=torch.float32, device=u.device)[None]
    robot.write_joint_state_to_sim(t, torch.zeros_like(t), joint_ids=arm_ids)
    for name, pose in places.items():
        u.scene[name].write_root_pose_to_sim(
            torch.tensor([pose], dtype=torch.float32, device=u.device))
    for _ in range(30):                      # let them settle onto the platform
        action = torch.zeros((1, 8), device=u.device)
        action[0, :7] = t[0]
        env.step(action, expensive=False)
    robot.write_joint_state_to_sim(t, torch.zeros_like(t), joint_ids=arm_ids)
    u.sim.render()
    u.scene.update(0)
    rgb = u.custom_render(True, transform_static=True)

    sim = np.asarray(rgb["external_cam"]).astype(np.uint8)
    real = np.asarray(Image.open(args_cli.frame).convert("RGB").resize(
        (sim.shape[1], sim.shape[0]), Image.LANCZOS))
    stem = Path(args_cli.frame).parent.parent.name
    Image.fromarray(sim).save(out / f"{stem}_sim.png")
    Image.fromarray(real).save(out / f"{stem}_real.png")
    Image.fromarray(np.concatenate([real, sim], 1)).save(out / f"{stem}_side.jpg", quality=93)
    blend = (real.astype(np.float32) * 0.5 + sim.astype(np.float32) * 0.5).astype(np.uint8)
    Image.fromarray(blend).save(out / f"{stem}_blend.jpg", quality=93)
    print(f"wrote {out}/{stem}_side.jpg")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
