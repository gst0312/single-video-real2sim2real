"""Put the simulated robot at a real recorded joint configuration and compare the pictures.

GSWorld recorded eight real-robot segments on 2026-07-10 with both DROID cameras and a
15 Hz joint-state log on the same wall clock (see docs/env_assets.md). That gives real
images at known joint angles, which is the only way to check the robot's pose, the camera
extrinsics and the scene together — our earlier check used a human demonstration frame,
where no joint angles exist.

For each requested frame this writes the recorded qpos straight into the articulation,
renders both cameras through PolaRiS's own path, and saves the real frame, the rendered
frame, and a 50/50 blend.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--segment", required=True, help="a directory such as .../20260710/home_static")
parser.add_argument("--third-serial", default="36087771")
parser.add_argument("--wrist-serial", default="16478870")
parser.add_argument("--frames", type=int, nargs="*", default=[100],
                    help="frame indices of the third-person camera")
parser.add_argument("--raytraced-robot", action="store_true",
                    help="draw the robot with the raytracer instead of PolaRiS's per-link splats")
parser.add_argument("--hide-objects", action="store_true",
                    help="drop the task objects out of sight; the real table was empty")
parser.add_argument("--condition", type=int, default=None,
                    help="place the objects at this initial condition instead of hiding them, "
                         "so the composite can be judged with the task objects in it")
parser.add_argument("--sample", type=int, default=None,
                    help="instead of --frames, take this many frames spread over the segment")
parser.add_argument("--report", default=None,
                    help="json with, per frame, the paired joint state and the errors")
parser.add_argument("--out-dir", required=True)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

from polaris.utils import load_eval_initial_conditions  # noqa: E402

import polaris_lfhv.environments  # noqa: E402,F401


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    segment = Path(args_cli.segment)
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    states = read_jsonl(segment / "robot_state.jsonl")
    state_t = np.array([s["t"] for s in states])
    state_q = np.array([s["qpos"] for s in states])
    cams = {"external_cam": args_cli.third_serial, "wrist_cam": args_cli.wrist_serial}
    stamps = {
        name: {int(r["i"]): r["t"] for r in read_jsonl(segment / f"cam{serial}" / "timestamps.jsonl")}
        for name, serial in cams.items()
    }
    print(f"{len(states)} joint states covering {state_t[-1] - state_t[0]:.1f} s")

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(args_cli.environment, cfg=env_cfg,
               robot_splat=not args_cli.raytraced_robot)
    env.reset()
    u = env.unwrapped

    # This script renders without stepping physics, so a physics-side pose write never
    # reaches the renderer: hidden objects would stay visible at their authored pose
    # (0, 0, -0.019) by the robot base, which the wrist camera picks up as a small
    # blue-and-yellow speck. Hide with USD visibility (verified; a fabric XFormPrim
    # move does not reach the RTX render), and for placed objects write the USD
    # transform directly alongside the physics pose (x y z qw qx qy qz).
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom
    stage = get_current_stage()

    if args_cli.hide_objects:
        for name in u.scene.rigid_objects:
            if name == "table_static":
                continue
            UsdGeom.Imageable(
                stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")).MakeInvisible()
    elif args_cli.condition is not None:
        usd = gym.spec(args_cli.environment).kwargs["usd_file"]
        _, conditions = load_eval_initial_conditions(usd)
        placed = conditions[args_cli.condition % len(conditions)]
        for name, pose in placed.items():
            u.scene[name].write_root_pose_to_sim(
                torch.tensor([pose], dtype=torch.float32, device=u.device))
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(v) for v in pose[:3]]))
            prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatf(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])))
        print(f"objects placed at initial condition {args_cli.condition}: {placed}")

    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]

    frames = args_cli.frames
    if args_cli.sample:
        # spread over the frames that actually fall inside the joint-state window
        inside = sorted(i for i, t in stamps["external_cam"].items()
                        if state_t[0] <= t <= state_t[-1])
        frames = [inside[int(round(r))]
                  for r in np.linspace(0, len(inside) - 1, args_cli.sample)]
        print(f"sampled frames {frames}")

    report = {"segment": segment.name, "frames": []}
    for index in frames:
        t = stamps["external_cam"].get(index)
        if t is None:
            print(f"frame {index}: no timestamp, skipping")
            continue
        k = int(np.argmin(np.abs(state_t - t)))
        gap = abs(state_t[k] - t)
        qpos = state_q[k]
        print(f"frame {index}: paired with joint state {k}, {gap * 1000:.0f} ms away")
        print(f"  qpos {np.round(qpos, 4)}")

        pos = torch.tensor(qpos, dtype=torch.float32, device=u.device)[None]
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=arm_ids)
        u.sim.render()
        u.scene.update(0)
        rgb = u.custom_render(True, transform_static=True)

        # the wrist camera runs at its own rate, so pair it by time as well
        wrist_index = min(stamps["wrist_cam"], key=lambda i: abs(stamps["wrist_cam"][i] - t))
        entry = {"frame": int(index), "state": int(k), "pair_gap_ms": float(gap * 1000),
                 "wrist_frame": int(wrist_index),
                 "wrist_pair_gap_ms": float(abs(stamps["wrist_cam"][wrist_index] - t) * 1000),
                 "qpos": [float(v) for v in qpos],
                 "achieved_qpos": [float(v) for v in
                                   robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()],
                 "cameras": {}}

        for cam, serial in cams.items():
            real_index = index if cam == "external_cam" else wrist_index
            real_path = segment / f"cam{serial}" / "rgb" / f"{real_index:06d}.png"
            if not real_path.exists():
                print(f"  {cam}: no real frame {real_path.name}")
                continue
            sim = np.asarray(rgb[cam]).astype(np.uint8)
            real = np.asarray(Image.open(real_path).convert("RGB").resize(
                (sim.shape[1], sim.shape[0]), Image.LANCZOS))
            stem = f"{segment.name}_{index:06d}_{cam}"
            Image.fromarray(sim).save(out / f"{stem}_sim.png")
            Image.fromarray(real).save(out / f"{stem}_real.png")
            Image.fromarray(np.concatenate([real, sim], axis=1)).save(out / f"{stem}_side.jpg", quality=92)
            blend = (real.astype(np.float32) * 0.5 + sim.astype(np.float32) * 0.5).astype(np.uint8)
            Image.fromarray(blend).save(out / f"{stem}_blend.jpg", quality=92)
            a, b = real.astype(np.float64), sim.astype(np.float64)
            mse = ((a - b) ** 2).mean()
            entry["cameras"][cam] = {
                "real_frame": int(real_index),
                "mae": float(np.abs(a - b).mean()),
                "psnr": float(10 * np.log10(255 ** 2 / mse)),
                "bright_ratio": float(b.mean() / a.mean()),
            }
            print(f"  {cam}: wrote {stem}_side.jpg, mae {entry['cameras'][cam]['mae']:.2f}")
        report["frames"].append(entry)

    if args_cli.report:
        with open(args_cli.report, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args_cli.report}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
