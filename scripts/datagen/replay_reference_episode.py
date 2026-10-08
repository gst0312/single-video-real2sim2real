"""Phase 2: replay a synthesised reference trajectory with physics, score it, render it.

This is the piece PolaRiS does not ship. The episode loop is
`polaris/scripts/eval.py` with one substitution: where eval asks a policy for an action,
this reads the next row of a reference trajectory from `synthesize_trajectories.py`. What
stays theirs is everything around it - `env.reset(object_positions=...)` places the objects
at the initial condition, `env.step(action, expensive=True)` advances the physics and
composites both camera views, and the rubric in `info["rubric"]` decides success.

Two things separate this from `preview_episode.py`, which writes joints and object poses
straight into the simulator: here the arm is *driven* by the 8-dimensional DROID action
(seven absolute joint positions plus the binary gripper) and the bottle is only moved by
being gripped. So the bottle can be missed, squeezed out, or dropped, and the rubric says
whether the pour actually happened. That is the phase 3 question - can the stock Robotiq
hold this bottle - and, once answered, the phase 4 data generator: episodes that pass are
exactly the ones whose observations become training data.

    scripts/polaris_env.sh python scripts/datagen/replay_reference_episode.py \
        --episode .../traj/episodes/cond000_ep02.npz --out-dir .../traj/rollouts --video

The horizon is set from the trajectory: PolaRiS's default is 30 s (450 steps) and a longer
episode would be truncated mid-replay, which teleports the arm home (measured 2026-08-12).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--episode", default=None, help="npz from synthesize_trajectories.py")
parser.add_argument("--episodes", default=None,
                    help="directory of episode npz to run in ONE process. Isaac takes 60-90 s "
                         "to boot and that is paid per process, so a hundred episodes launched "
                         "one at a time spend more than an hour just starting up. The "
                         "environment is built once here and reset per episode; fan out across "
                         "GPUs by giving each process a different slice (--shard i/n).")
parser.add_argument("--shard", default="0/1",
                    help="i/n: take every n-th episode starting at i, for running n processes")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--video", action="store_true", help="write an mp4 of both views")
parser.add_argument("--video-scale", type=float, default=0.5)
parser.add_argument("--save-obs", action="store_true",
                    help="also store the 180x320 observations phase 4 needs for the dataset")
parser.add_argument("--settle", type=int, default=5,
                    help="steps held at the reset pose before the trajectory starts")
parser.add_argument("--hold", type=int, default=20,
                    help="steps holding the final pose, so a late slip still shows up")
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

OBS_SIZE = (320, 180)   # width, height: the official cotrain dataset's image size


def run_episode(env, u, path, longest_rate):
    """One episode on an already-built environment. Returns the report dict."""
    d = np.load(path)
    qpos, grip = d["qpos"], d["gripper"]
    name = Path(path).stem
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = len(qpos) + args_cli.hold
    print(f"{name}: {len(qpos)} reference steps, condition {int(d['condition_id'])}, "
          f"time scale {float(d['time_scale']):g}, ramp {int(d['n_ramp'])}")

    conditions = json.load(open(Path(u.usd_file).parent / "initial_conditions.json"))
    condition = conditions["poses"][int(d["condition_id"])]
    obs, info = env.reset(object_positions=condition)
    print(f"instruction: {conditions['instruction']!r}")

    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]

    action = torch.zeros((1, 8), device=u.device)
    action[0, :7] = torch.tensor(qpos[0], dtype=torch.float32, device=u.device)
    for _ in range(args_cli.settle):
        obs, *_ = env.step(action, expensive=False)

    writer = None
    if args_cli.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out / f"{name}.mp4", fps=int(d["rate_hz"]),
                                    quality=7, macro_block_size=8)

    frames = {"external_cam": [], "wrist_cam": []} if args_cli.save_obs else None
    render = args_cli.video or args_cli.save_obs
    commanded, commanded_grip = [], []
    achieved, grip_state, bottle_z, ee_dist, success, progress = [], [], [], [], [], []
    cup_pose, bottle_pose = [], []
    for t in range(steps):
        k = min(t, len(qpos) - 1)            # hold the last pose for the tail
        action[0, :7] = torch.tensor(qpos[k], dtype=torch.float32, device=u.device)
        action[0, 7] = float(grip[k])
        obs, rew, term, trunc, info = env.step(action, expensive=render)

        commanded.append(qpos[k].copy())
        commanded_grip.append(float(grip[k]))
        achieved.append(obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy().copy())
        grip_state.append(float(obs["policy"]["gripper_pos"][0]))
        bottle_z.append(float(u.scene["mustard"].data.root_pos_w[0, 2]))
        # the cup is a physics body too: an arm that clips it moves the pour target, which
        # would otherwise show up only as an unexplained rubric failure
        cup_pose.append(np.concatenate([
            u.scene["blue_cup"].data.root_pos_w[0].detach().cpu().numpy(),
            u.scene["blue_cup"].data.root_quat_w[0].detach().cpu().numpy()]))
        # what the bottle actually did, against what the reference said it would: the gap
        # between them is the rigid-follow assumption failing, and it is the thing a
        # kinematic pipeline cannot see
        bottle_pose.append(np.concatenate([
            u.scene["mustard"].data.root_pos_w[0].detach().cpu().numpy(),
            u.scene["mustard"].data.root_quat_w[0].detach().cpu().numpy()]))
        # the same pair `checkers.reach` compares, logged so its threshold can be measured
        ee_dist.append(float(torch.norm(u.scene["mustard"].data.root_pos_w[0]
                                        - u.scene["ee_frame"].data.target_pos_w[0, 0])))
        success.append(bool(info["rubric"]["success"]))
        progress.append(float(info["rubric"]["progress"]))
        if not render:
            continue

        views = [np.asarray(obs["splat"][c]).astype(np.uint8)
                 for c in ("external_cam", "wrist_cam")]
        if frames is not None:
            for cam, img in zip(("external_cam", "wrist_cam"), views):
                frames[cam].append(np.asarray(
                    Image.fromarray(img).resize(OBS_SIZE, Image.LANCZOS)))
        if writer is not None:
            panel = np.concatenate(views, axis=1)
            size = (int(panel.shape[1] * args_cli.video_scale),
                    int(panel.shape[0] * args_cli.video_scale))
            writer.append_data(np.asarray(
                Image.fromarray(panel).resize(size, Image.LANCZOS)))
    if writer is not None:
        writer.close()

    commanded, achieved = np.array(commanded), np.array(achieved)
    err = np.degrees(achieved - commanded)
    report = {
        "episode": name,
        "condition": int(d["condition_id"]),
        "time_scale": float(d["time_scale"]),
        "steps": int(steps),
        "success": bool(np.any(success)),
        "success_step": int(np.argmax(success)) if np.any(success) else -1,
        "progress_max": float(np.max(progress)),
        "tracking_rms_deg": float(np.sqrt((err ** 2).mean())),
        "tracking_max_deg": float(np.abs(err).max()),
        "bottle_z_start": bottle_z[0],
        "bottle_z_max": float(np.max(bottle_z)),
        "bottle_z_end": bottle_z[-1],
        "bottle_lift": float(np.max(bottle_z) - bottle_z[0]),
        "ee_to_bottle_min": float(np.min(ee_dist)),
        "cup_moved_m": float(np.linalg.norm(np.array(cup_pose)[:, :3]
                                            - np.array(cup_pose)[0, :3], axis=1).max()),
    }
    # how far the physical bottle drifted from the trajectory the generator assumed it
    # would follow; index k of the reference lines up with step k of the replay
    bp = np.array(bottle_pose)
    ref = d["mustard_traj"][:len(bp)]
    gap = np.linalg.norm(bp[:len(ref), :3] - ref[:, :3], axis=1)
    grasped = slice(int(d["close_step"]) + 10, len(gap))
    report["bottle_vs_reference"] = {
        "gap_at_grasp_m": float(gap[int(d["close_step"]) + 10]) if grasped.start < len(gap) else None,
        "gap_max_m": float(gap[grasped].max()) if grasped.start < len(gap) else None,
        "gap_end_m": float(gap[-1]),
    }
    np.savez(out / f"{name}_bottle_track.npz", actual=bp, reference=ref,
             cup=np.array(cup_pose))
    # why the pour criterion did or did not fire, in its own terms
    for c in getattr(u.rubric, "criteria", []):
        fn = c[0] if isinstance(c, tuple) else c
        if hasattr(fn, "stats"):
            report["pour_closest"] = fn.stats()
            tr = np.array(fn.trace()).reshape(-1, 3)
            if len(tr):
                r = report["pour_closest"]["cup_radius"]
                ok_t, ok_d, ok_a = tr[:, 0] > 60, tr[:, 1] < r, (tr[:, 2] > 0) & (tr[:, 2] < 0.30)
                report["pour_steps"] = {
                    "evaluated": int(len(tr)),
                    "tilt_ok": int(ok_t.sum()), "over_cup_ok": int(ok_d.sum()),
                    "above_rim_ok": int(ok_a.sum()),
                    "tilt_and_over_cup": int((ok_t & ok_d).sum()),
                    "all_three": int((ok_t & ok_d & ok_a).sum()),
                }
                np.save(out / f"{name}_pour_trace.npy", tr)
    with open(out / f"{name}.json", "w") as f:
        json.dump(report, f, indent=1)
    if frames is not None:
        np.savez_compressed(
            out / f"{name}_obs.npz",
            external_cam=np.array(frames["external_cam"], dtype=np.uint8),
            wrist_cam=np.array(frames["wrist_cam"], dtype=np.uint8),
            joint_position=achieved, gripper_position=np.array(grip_state),
            action=np.concatenate([commanded, np.array(commanded_grip)[:, None]], axis=1),
            instruction=conditions["instruction"])
    print(f"success {report['success']} (progress {report['progress_max']:.2f}), "
          f"bottle lifted {report['bottle_lift'] * 100:.1f} cm, "
          f"tracking rms {report['tracking_rms_deg']:.2f} deg, "
          f"ee-to-bottle min {report['ee_to_bottle_min']:.3f} m")
    print("  bottle vs reference: " + ", ".join(
        f"{k} {v:.3f}" for k, v in report["bottle_vs_reference"].items() if v is not None)
        + f"; cup moved {report['cup_moved_m']:.3f} m")
    if "pour_closest" in report:
        print("  pour closest approach: " + ", ".join(
            f"{k} {v:.3f}" for k, v in report["pour_closest"].items() if v is not None))
    print(f"wrote {out}/{name}.json" + (f" and {name}.mp4" if args_cli.video else ""))
    return report


def main():
    if (args_cli.episode is None) == (args_cli.episodes is None):
        raise SystemExit("pass exactly one of --episode or --episodes")
    if args_cli.episode:
        paths = [args_cli.episode]
    else:
        paths = sorted(p for p in Path(args_cli.episodes).glob("*.npz")
                       if not p.name.endswith(("_obs.npz", "_bottle_track.npz")))
        i, n = (int(v) for v in args_cli.shard.split("/"))
        paths = [str(p) for p in paths[i::n]]
    if not paths:
        raise SystemExit("no episodes to run")

    # one environment for all of them; the horizon has to cover the longest episode,
    # because PolaRiS's default 30 s would truncate mid-replay and teleport the arm home
    lengths = [(len(np.load(p)["qpos"]) + args_cli.hold, float(np.load(p)["rate_hz"]))
               for p in paths]
    longest, rate = max(lengths)
    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = (longest + args_cli.settle + 50) / rate
    env = gym.make(args_cli.environment, cfg=env_cfg)
    u = env.unwrapped
    print(f"{len(paths)} episodes in this process, horizon {env_cfg.episode_length_s:.0f} s")

    for k, path in enumerate(paths):
        print(f"--- [{k + 1}/{len(paths)}] {Path(path).stem}")
        run_episode(env, u, path, rate)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
