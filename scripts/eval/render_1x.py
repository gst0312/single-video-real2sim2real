"""Re-run an evaluation episode rendering every step, so the video plays at real speed.

PolaRiS's evaluation stores one video frame per policy inference and writes the file at
15 fps, so with an open-loop horizon of 8 the result is 8x faster than the arm actually
moves - 1050 simulated steps become an 8.8 second clip. That is fine for checking whether a
rollout succeeded and misleading for judging how fast the motion is.

Two things have to change together to get real speed. The client only returns a frame on the
steps where it re-infers, so `infer` is called with `return_viz=True` to get one every step;
and the environment only re-renders when the client asks it to (`rerender`), so that is
forced on as well - without it every frame in a chunk would be the same stale image and the
clip would be temporally right but visually stuttering at 1.9 Hz.

Rendering every step instead of every eighth costs roughly eight times the wall clock, which
is why this is for the handful of clips worth showing, not for scoring.

    scripts/polaris_env.sh python scripts/eval/render_1x.py \
        --policy.client DroidJointPos --policy.port 8009 --policy.open-loop-horizon 8 \
        --environment DROID-PourMustard \
        --initial-conditions-file <one-condition.json> --rollouts 1 --run-folder <dir>
"""

import os
import sys
from pathlib import Path

POLARIS = Path(os.environ.get("POLARIS_ROOT", "polaris"))
sys.path.insert(0, str(POLARIS / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaaclab.app as _app  # noqa: E402
from isaaclab.app import AppLauncher as _Launcher  # noqa: E402

from eval_policy import EVAL_HORIZON_S  # noqa: E402


class _RegisterOurs(_Launcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import polaris_lfhv.environments  # noqa: F401

        import isaaclab_tasks.utils as _tasks

        _parse = _tasks.parse_env_cfg

        def parse_env_cfg(*a, **k):
            cfg = _parse(*a, **k)
            cfg.episode_length_s = EVAL_HORIZON_S
            return cfg

        _tasks.parse_env_cfg = parse_env_cfg


_app.AppLauncher = _RegisterOurs

import tyro  # noqa: E402
from polaris.config import EvalArgs  # noqa: E402


def main():
    args = tyro.cli(EvalArgs)
    import eval as polaris_eval  # noqa: E402

    from polaris.policy.droid_jointpos_client import DroidJointPosClient

    original = DroidJointPosClient.infer

    def infer(self, obs, instruction, return_viz=False):
        # ignore the caller's choice: eval.py never asks for a frame on the steps between
        # inferences, and those are exactly the ones missing from the clip
        return original(self, obs, instruction, return_viz=True)

    DroidJointPosClient.infer = infer
    # `rerender` decides whether env.step does the expensive 3DGS render; on by default it is
    # only true when a new chunk is due, which would leave the extra frames stale
    DroidJointPosClient.rerender = property(lambda self: True)

    print(f"[render_1x] rendering every step; the video will run {EVAL_HORIZON_S:.0f} s at real speed")
    polaris_eval.main(args)


if __name__ == "__main__":
    main()
