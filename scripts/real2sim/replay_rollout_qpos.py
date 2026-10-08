"""Replay a recorded joint trajectory through the environment and measure the tracking error.

`compare_to_real_rollout.py` writes the recorded angles straight into the articulation, which
answers "does the picture match" but says nothing about whether the environment can *drive*
the robot along a real trajectory. Phase 4 replays reference trajectories with physics on and
keeps only the episodes that behave, so the tracking error of the controller is a number that
pipeline needs first.

The action space is PolaRiS's DROID one: absolute joint positions, seven arm joints plus the
gripper, stepped at the environment's own rate. The recorded segments log qpos at about 15 Hz,
the same rate, so each recorded state is issued as one action and what the simulator reaches
by the next step is compared against it.

    scripts/polaris_env.sh python scripts/real2sim/replay_rollout_qpos.py \
        --segment .../20260710/randomwalk_2 --report .../replay_randomwalk_2.json

Two sources are accepted. --segment is a GSWorld rollout directory (robot_state.jsonl, arm
only, gripper stays open). --npz is an episode exported by `export_lerobot_qpos.py` from the
2026-02-09 teleop dataset: there the measured qpos stream is replayed as the absolute
position action and the recorded continuous gripper command is fed with it, so pick-and-pour
episodes close the gripper at the recorded times. Those episodes were driven on the real
robot by joint *velocity* commands; replaying their measured positions through the position
head and comparing what the simulator reaches against what the real arm reached is exactly
the check that this action head does not need the actuator identification the velocity head
demanded. --hide-objects sinks the task objects (same mechanism as compare_to_real_rollout)
so a trajectory recorded with other table dressing is not judged on collisions with ours.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--segment", default=None, help="GSWorld rollout directory")
parser.add_argument("--npz", default=None, help="episode npz from export_lerobot_qpos.py")
parser.add_argument("--hide-objects", action="store_true",
                    help="sink every rigid object except the scene before replaying")
parser.add_argument("--stride", type=int, default=1, help="use every Nth recorded state")
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument("--settle", type=int, default=5, help="steps held at the first pose")
parser.add_argument("--report", required=True)
parser.add_argument("--trace", default=None,
                    help="also dump per-step commanded/achieved qpos and the end-effector "
                         "world position to this npz, for localising where an error grows")
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

import r2s2r.environments  # noqa: E402,F401


def main():
    if (args_cli.segment is None) == (args_cli.npz is None):
        raise SystemExit("pass exactly one of --segment or --npz")
    grip = None
    task = None
    if args_cli.npz:
        d = np.load(args_cli.npz)
        source = Path(args_cli.npz).stem
        task = str(d["task"])
        q = d["qpos"][:, :7]
        # the recorded continuous gripper command; the environment's own client-side
        # convention binarises at 0.5, intermediate widths are not reproducible
        grip = d["actions"][:, 7]
        grip_measured = np.asarray(d["gripper"]).reshape(-1)
        rate = float(d["fps"])
        q, grip, grip_measured = (a[::args_cli.stride][:args_cli.max_steps]
                                  for a in (q, grip, grip_measured))
        print(f"{len(q)} of {len(d['qpos'])} frames at {rate:.1f} Hz, task '{task}'")
    else:
        segment = Path(args_cli.segment)
        states = [json.loads(line) for line in open(segment / "robot_state.jsonl")
                  if line.strip()]
        source = segment.name
        t = np.array([s["t"] for s in states])
        q = np.array([s["qpos"] for s in states])[:, :7]
        q = q[::args_cli.stride][:args_cli.max_steps]
        rate = (len(states) - 1) / (t[-1] - t[0])
        print(f"{len(states)} states at {rate:.1f} Hz, replaying {len(q)} of them")

    env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
    # PolaRiS's horizon is 30 s (450 steps); a longer recording would hit it mid-replay and
    # the auto-reset teleports the arm home, which reads as a huge one-off tracking error
    # (measured: j7 snapping from -139.6 deg to 0 at step 447 of a 788-step episode).
    env_cfg.episode_length_s = 1e6
    env = gym.make(args_cli.environment, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    arm_ids = [robot.data.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    grip_id = robot.data.joint_names.index("finger_joint")

    if args_cli.hide_objects:
        # same mechanism as compare_to_real_rollout.py: sink everything but the scene
        for name in u.scene.rigid_objects:
            if name == "table_static":
                continue
            pose = torch.tensor([[0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0]], device=u.device)
            u.scene[name].write_root_pose_to_sim(pose)

    # start on the first recorded pose so the first action is not a jump
    start = torch.tensor(q[0], dtype=torch.float32, device=u.device)[None]
    robot.write_joint_state_to_sim(start, torch.zeros_like(start), joint_ids=arm_ids)
    action = torch.zeros((1, 8), device=u.device)
    action[0, :7] = start[0]
    for _ in range(args_cli.settle):
        env.step(action, expensive=False)

    # the deepest arm body carried in the articulation; the flange and gripper hang off it
    ee_body = robot.data.body_names.index(
        "panda_link8" if "panda_link8" in robot.data.body_names else "panda_link7")
    commanded, achieved, grip_achieved, ee_pos = [], [], [], []
    for k, target in enumerate(q):
        action[0, :7] = torch.tensor(target, dtype=torch.float32, device=u.device)
        if grip is not None:
            action[0, 7] = float(grip[k])
        env.step(action, expensive=False)
        commanded.append(target.copy())
        achieved.append(robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy())
        grip_achieved.append(float(robot.data.joint_pos[0, grip_id].detach().cpu()))
        ee_pos.append(robot.data.body_pos_w[0, ee_body].detach().cpu().numpy().copy())
    commanded = np.array(commanded)
    achieved = np.array(achieved)

    err = np.degrees(achieved - commanded)
    report = {
        "source": source,
        "steps": int(len(q)),
        "record_rate_hz": float(rate),
        "per_joint_rms_deg": [float(v) for v in np.sqrt((err ** 2).mean(0))],
        "per_joint_max_deg": [float(v) for v in np.abs(err).max(0)],
        "rms_deg": float(np.sqrt((err ** 2).mean())),
        "max_deg": float(np.abs(err).max()),
        "commanded_range_deg": [float(v) for v in np.degrees(commanded.max(0) - commanded.min(0))],
    }
    if task is not None:
        report["task"] = task
    if grip is not None:
        # both closures on 0..1: measured on the real gripper, the sim finger joint's
        # 0..pi/4 range normalised. The sim gripper is binary, so intermediate widths
        # differ by construction; what should agree is which steps are open vs closed.
        sim_closure = np.array(grip_achieved) / (np.pi / 4)
        agree = (sim_closure > 0.5) == (grip_measured > 0.5)
        report["gripper"] = {
            "binary_agreement": float(agree.mean()),
            "closure_mae": float(np.abs(sim_closure - grip_measured).mean()),
        }
    if args_cli.trace:
        np.savez(args_cli.trace, commanded=commanded, achieved=achieved,
                 gripper_achieved=np.array(grip_achieved), ee_pos=np.array(ee_pos))
        print(f"wrote trace {args_cli.trace}")
    print(f"tracking error: rms {report['rms_deg']:.3f} deg, max {report['max_deg']:.3f} deg")
    print("  per joint rms " + " ".join(f"{v:.3f}" for v in report["per_joint_rms_deg"]))
    with open(args_cli.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args_cli.report}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
