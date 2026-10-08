"""Phase 1 acceptance: can our environment be made, reset, stepped and rendered?

Mirrors the minimal example in the polaris README, but for our environment, and saves the
two rendered views so they can be compared against a real photograph from the same
viewpoint (the external camera is placed from the ZED calibration, so the comparison is
what tells us the scene is aligned).
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--environment", default="DROID-PourMustard")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--condition", type=int, default=None,
                    help="index into initial_conditions.json; without it the objects stay at "
                         "the poses scene.usda gives them")
parser.add_argument("--out-prefix", default="/tmp/pour_mustard")
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

env_cfg = parse_env_cfg(args_cli.environment, device="cuda", num_envs=1, use_fabric=True)
env = gym.make(args_cli.environment, cfg=env_cfg)

object_positions = {}
if args_cli.condition is not None:
    usd = gym.spec(args_cli.environment).kwargs["usd_file"]
    instruction, conditions = load_eval_initial_conditions(usd)
    object_positions = conditions[args_cli.condition % len(conditions)]
    print(f"instruction: {instruction}")
    print(f"initial condition {args_cli.condition}: {object_positions}", flush=True)

obs, info = env.reset(object_positions=object_positions)
print("reset ok", flush=True)
for name, value in obs.items():
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {name}/{k}: {getattr(v, 'shape', type(v))}")

u = env.unwrapped
print("rigid objects in scene:", list(u.scene.rigid_objects.keys()))
print("cameras in scene:", [k for k in u.scene.sensors])
print("splats loaded:", list(u.splat_renderer.pcds.keys()))

def object_poses():
    out = {}
    for name in u.scene.rigid_objects:
        s = u.scene[name].data.root_state_w[0]
        out[name] = (s[:3].detach().cpu().numpy().copy(), s[3:7].detach().cpu().numpy().copy())
    return out


before = object_poses()
t0 = time.time()
for i in range(args_cli.steps):
    action = torch.zeros((1, 8))
    action[0, :7] = torch.tensor(u.scene["robot"].data.default_joint_pos[0, :7])
    obs, rew, term, trunc, info = env.step(action, expensive=True)
dt = time.time() - t0
print(f"stepped {args_cli.steps} times ok, {dt:.1f} s, {args_cli.steps / dt:.2f} steps/s "
      "(with rendering)")

# Objects are dropped a millimetre above the collision platform and should settle, not
# sink, drift or topple. Tilt is the angle between the body's z axis and the world's.
after = object_poses()
for name in before:
    p0, p1 = before[name][0], after[name][0]
    w, x, y, z = after[name][1]
    zaxis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    tilt = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
    print(f"  {name}: z {p0[2]:+.4f} -> {p1[2]:+.4f} m, moved "
          f"{np.linalg.norm(p1 - p0) * 1000:5.1f} mm, tilt {tilt:5.1f} deg")

for cam, img in obs["splat"].items():
    path = f"{args_cli.out_prefix}_{cam}.png"
    Image.fromarray(np.asarray(img)).save(path)
    print("wrote", path)

env.close()
simulation_app.close()
