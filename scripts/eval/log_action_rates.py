"""Record what joint velocities the policy actually commands, and check them against FR3.

The rollout videos are 8x faster than real time - PolaRiS's eval stores one frame per policy
inference, and the open-loop horizon is 8 - so they say nothing about how fast the arm really
moves. The training data was generated against the FR3 velocity envelope by construction, but
the policy's output is its own, and nothing between the policy and the robot enforces a
velocity limit. Before the real robot, the commanded rate has to be measured.

This runs PolaRiS's evaluation unchanged and records every action the client hands the
environment, then reports the implied joint velocity: consecutive actions are one control
period apart (15 Hz), and the action is an absolute joint position, so the velocity is the
difference over that period.

    scripts/polaris_env.sh python scripts/eval/log_action_rates.py \
        --policy.client DroidJointPos --policy.port 8009 --policy.open-loop-horizon 8 \
        --environment DROID-PourMustard \
        --initial-conditions-file <conditions.json> --rollouts 2 --run-folder <dir>

    python scripts/eval/log_action_rates.py --analyse <dir>/actions.npz

The second form re-reads a recording and needs no simulator, which is why the Isaac imports
live inside main() rather than at the top.
"""

import os
import sys
from pathlib import Path

import numpy as np

POLARIS = Path(os.environ.get("POLARIS_ROOT", "polaris"))
sys.path.insert(0, str(POLARIS / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CONTROL_HZ = 15.0

#: Episode horizon used by the evaluation, in seconds; see scripts/eval/eval_policy.py.
EVAL_HORIZON_S = 70.0

#: The ceiling the trajectory synthesiser was held to: FR3's own limit intersected with the
#: deployment soft-safety wall. Same numbers as scripts/datagen/synthesize_trajectories.py.
FR3_V_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
DEPLOY_V_SOFT = np.array([1.575] * 4 + [2.01] * 3)
VEL_CEILING = np.minimum(FR3_V_MAX, DEPLOY_V_SOFT)


def _wrap(launcher_cls):
    """PolaRiS's AppLauncher with our environment and horizon, as in eval_policy.py."""

    class _RegisterOurs(launcher_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import polaris_lfhv.environments  # noqa: F401

            import isaaclab_tasks.utils as _tasks
            from eval_policy import EVAL_HORIZON_S

            _parse = _tasks.parse_env_cfg

            def parse_env_cfg(*a, **k):
                cfg = _parse(*a, **k)
                cfg.episode_length_s = EVAL_HORIZON_S
                return cfg

            _tasks.parse_env_cfg = parse_env_cfg

    return _RegisterOurs


def main():
    import isaaclab.app as _app
    from isaaclab.app import AppLauncher

    _app.AppLauncher = _wrap(AppLauncher)

    import tyro
    from polaris.config import EvalArgs

    args = tyro.cli(EvalArgs)

    import eval as polaris_eval
    from polaris.policy.droid_jointpos_client import DroidJointPosClient

    recorded = []
    original = DroidJointPosClient.infer
    out = Path(args.run_folder) / "actions.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    def infer(self, obs, instruction, return_viz=False):
        action, viz = original(self, obs, instruction, return_viz)
        recorded.append(np.asarray(action, dtype=np.float64).copy())
        # write as we go: PolaRiS's eval ends with simulation_app.close(), which takes the
        # process down hard, so anything saved only at the end is never saved at all
        if len(recorded) % 100 == 0:
            np.savez_compressed(out, action=np.stack(recorded))
        return action, viz

    DroidJointPosClient.infer = infer
    polaris_eval.main(args)


def report(a, episode_len=None):
    """Joint velocity implied by consecutive absolute-position commands.

    The recording is every episode back to back, so the step across an episode boundary is
    not one control period - it is the reset that puts the arm back at its home pose. Left
    in, that single sample reads as a 19x overspeed and is pure artefact, so the boundaries
    are cut before differencing.
    """
    if episode_len is None:
        # every episode runs the full horizon: the evaluation does not stop early on success
        episode_len = int(EVAL_HORIZON_S * CONTROL_HZ)
    segments = [a[i : i + episode_len] for i in range(0, len(a), episode_len)]
    segments = [s for s in segments if len(s) > 1]
    v = np.concatenate([np.abs(np.diff(s[:, :7], axis=0)) for s in segments]) * CONTROL_HZ
    ratio = v / VEL_CEILING
    print(f"cut into {len(segments)} episodes of {episode_len} steps, skipping the reset jump between episodes")

    print(f"{len(a)} recorded actions, {len(v)} velocity samples")
    print(f"velocity ceiling (FR3 vs deployment soft wall, tighter): {np.array2string(VEL_CEILING, precision=2)}")
    print(f"peak joint velocity rad/s:  {np.array2string(v.max(axis=0), precision=2)}")
    print(f"fraction of the ceiling:    {np.array2string(ratio.max(axis=0), precision=2)}")
    print(f"overall peak {ratio.max():.2f} of the ceiling (joint {int(ratio.max(axis=0).argmax()) + 1})")
    over = int((ratio > 1.0).any(axis=1).sum())
    print(f"steps over the limit: {over} / {len(v)} ({100 * over / len(v):.1f}%)")
    if over:
        print(f"  worst offending step is {ratio.max(axis=1).max():.2f}x the ceiling")
    # the gripper column should stay binary: that is what the data holds and what the sim
    # client enforces, and it is why binarising on the real robot is lossless
    print(f"gripper values seen: {np.unique(a[:, 7])}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--analyse":
        report(np.load(sys.argv[2])["action"])
    else:
        main()
