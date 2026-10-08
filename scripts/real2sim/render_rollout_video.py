"""Render a whole recorded segment and write the real-vs-sim video for it.

The same idea as the still comparisons in `compare_to_real_rollout.py`, run over every
frame instead of a handful: at each step the recorded joint angles go into the articulation
and both cameras are rendered, then each frame is written straight to the video file, so
memory does not grow with the length of the segment. GSWorld ships the same artefact for
their own scenes (`docs/renders/scene06/real_vs_sim/<seg>_<cam>_real_sim.mp4`), which is
what makes the two comparable.

    scripts/polaris_env.sh python scripts/real2sim/render_rollout_video.py \
        --segment .../20260710/randomwalk_2 --stride 2 --out-dir .../videos

`--npz` renders a teleop episode exported by `export_lerobot_qpos.py --with-images`
instead of a recorded segment: frames pair with states by index (same 15 Hz stream, no
timestamp matching needed), and the measured continuous gripper closure is written into
the finger joint and its follower joints so the hand closes like the real one did.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--segment", default=None)
parser.add_argument("--npz", default=None,
                    help="teleop episode from export_lerobot_qpos.py --with-images")
parser.add_argument("--third-serial", default="36087771")
parser.add_argument("--wrist-serial", default="16478870")
parser.add_argument("--stride", type=int, default=2, help="render every Nth camera frame")
parser.add_argument("--max-frames", type=int, default=600)
parser.add_argument("--scale", type=float, default=0.5, help="output size per panel")
parser.add_argument("--blend", action="store_true", help="add a third panel, real over sim")
parser.add_argument("--hide-objects", action="store_true", default=True)
parser.add_argument("--out-dir", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

import r2s2r.environments  # noqa: E402,F401


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


#: the Robotiq follower joints and their sign relative to finger_joint, for writing the
#: measured closure kinematically. MEASURED, not assumed: the env's own physics was
#: commanded closed and the settled joints read back (finger_joint +0.7854 gives
#: right_outer +0.7854, left_inner_finger -0.7854, right_inner_finger +0.7854, both
#: inner_finger_knuckles -0.7854); a guessed sign set rendered scissored fingers.
GRIPPER_FOLLOWERS = (("right_outer_knuckle_joint", 1.0),
                     ("left_inner_finger_joint", -1.0),
                     ("right_inner_finger_joint", 1.0),
                     ("left_inner_finger_knuckle_joint", -1.0),
                     ("right_inner_finger_knuckle_joint", -1.0))


def main():
    assert (args_cli.segment is None) != (args_cli.npz is None), \
        "pass exactly one of --segment / --npz"
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cams = {"external_cam": args_cli.third_serial, "wrist_cam": args_cli.wrist_serial}

    if args_cli.segment:
        segment = Path(args_cli.segment)
        stem = segment.name
        states = read_jsonl(segment / "robot_state.jsonl")
        state_t = np.array([s["t"] for s in states])
        state_q = np.array([s["qpos"] for s in states])[:, :7]
        stamps = {name: {int(r["i"]): r["t"]
                         for r in read_jsonl(segment / f"cam{serial}" / "timestamps.jsonl")}
                  for name, serial in cams.items()}
        inside = sorted(i for i, t in stamps["external_cam"].items()
                        if state_t[0] <= t <= state_t[-1])
        frames = inside[::args_cli.stride][:args_cli.max_frames]
        rate = (len(states) - 1) / (state_t[-1] - state_t[0])
        fps = rate / args_cli.stride
        grip = None
    else:
        ep = np.load(args_cli.npz)
        stem = f"hf_{int(ep['episode_index']):06d}"
        state_q = ep["qpos"][:, :7]
        grip = ep["gripper"].reshape(-1)
        reals = {"external_cam": ep["ext_images"], "wrist_cam": ep["wrist_images"]}
        frames = list(range(len(state_q)))[::args_cli.stride][:args_cli.max_frames]
        fps = float(ep["fps"]) / args_cli.stride
        print(f"episode task: {ep['task']}")
    print(f"{len(frames)} frames at {fps:.1f} fps")

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    if args_cli.hide_objects:
        # Hide at render level, not by moving the bodies: this script never steps
        # physics, so a physics-side pose write never reaches the renderer and the
        # raytraced meshes would stay visible at their authored pose (0, 0, -0.019)
        # by the robot base - the wrist camera picks them up as a tiny blue-and-yellow
        # speck whenever it looks toward the base (verified; a fabric XFormPrim move
        # does not reach the RTX render either, USD visibility does).
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        stage = get_current_stage()
        for name in u.scene.rigid_objects:
            if name != "table_static":
                UsdGeom.Imageable(
                    stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")).MakeInvisible()
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    grip_ids, grip_mult = [robot.data.joint_names.index("finger_joint")], [1.0]
    for name, mult in GRIPPER_FOLLOWERS:
        if name in robot.data.joint_names:
            grip_ids.append(robot.data.joint_names.index(name))
            grip_mult.append(mult)
    grip_mult_t = torch.tensor(grip_mult, dtype=torch.float32, device=u.device)

    writers = {name: imageio.get_writer(out / f"{stem}_{name}_real_sim.mp4",
                                        fps=max(1, round(fps)), quality=7, macro_block_size=8)
               for name in cams}
    t0 = time.time()
    for n, index in enumerate(frames):
        if args_cli.segment:
            t = stamps["external_cam"][index]
            k = int(np.argmin(np.abs(state_t - t)))
            wrist_index = min(stamps["wrist_cam"], key=lambda i: abs(stamps["wrist_cam"][i] - t))
        else:
            k = index
        pos = torch.tensor(state_q[k], dtype=torch.float32, device=u.device)[None]
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=arm_ids)
        if grip is not None:
            theta = float(np.clip(grip[k], 0.0, 1.0)) * 0.7854
            gpos = (grip_mult_t * theta)[None]
            robot.write_joint_state_to_sim(gpos, torch.zeros_like(gpos), joint_ids=grip_ids)
        u.sim.render()
        u.scene.update(0)
        rgb = u.custom_render(True, transform_static=True)

        for cam, serial in cams.items():
            if args_cli.segment:
                real_path = (segment / f"cam{serial}" / "rgb" /
                             f"{(index if cam == 'external_cam' else wrist_index):06d}.png")
                if not real_path.exists():
                    continue
                real_img = Image.open(real_path).convert("RGB")
            else:
                real_img = Image.fromarray(reals[cam][k])
            sim = np.asarray(rgb[cam]).astype(np.uint8)
            h, w = sim.shape[:2]
            size = (int(w * args_cli.scale), int(h * args_cli.scale))
            real = np.asarray(real_img.resize(size, Image.LANCZOS))
            sim_s = np.asarray(Image.fromarray(sim).resize(size, Image.LANCZOS))
            panels = [real, np.full((size[1], 4, 3), 255, np.uint8), sim_s]
            if args_cli.blend:
                panels += [np.full((size[1], 4, 3), 255, np.uint8),
                           (0.5 * real.astype(np.float32)
                            + 0.5 * sim_s.astype(np.float32)).astype(np.uint8)]
            writers[cam].append_data(np.concatenate(panels, axis=1))
        if n % 50 == 0:
            print(f"  {n}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)

    for w_ in writers.values():
        w_.close()
    print(f"wrote {len(frames)} frames in {time.time() - t0:.0f} s to {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
