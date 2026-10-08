"""Run PolaRiS's evaluation on our environment.

Simulation evaluation is `polaris/scripts/eval.py` plus its `DroidJointPos` client with
zero new code, and that holds: the episode loop, the rubric handling, the
CSV and the video all come from their script, untouched. What is missing is only that their
script imports `polaris.environments`, which registers their six environments and not ours,
so `parse_env_cfg("DROID-PourMustard")` raises NameNotFound before anything else happens.

Registering ours is awkward for one reason: Isaac has to be running before `r2s2r
.environments` can be imported (it pulls in isaaclab), and their script launches Isaac
inside `main`, then immediately uses the gym registry. The only seam between those two is
the AppLauncher itself, so that is what gets wrapped - the app comes up exactly as before,
and our `gym.register` runs on the next line. Nothing in their evaluation loop is touched.

    scripts/polaris_env.sh python scripts/eval/eval_policy.py \
        --policy.client DroidJointPos --policy.port 8000 --policy.open-loop-horizon 8 \
        --environment DROID-PourMustard \
        --initial-conditions-file <PolaRiS-Hub>/pour_mustard/eval_conditions.json \
        --rollouts 30 --run-folder <where to write>

Use `eval_conditions.json`, not the environment's own `initial_conditions.json`: the latter
is what the training data was generated from, and scoring a policy on the layouts it was
trained on says nothing about whether it generalises.

The policy has to be served first, and its log must show the norm stats coming from the
base checkpoint's official assets - the same check `docs/real_robot.md` puts before
touching the real robot.
"""

import os
import sys
from pathlib import Path

POLARIS = Path(os.environ.get("POLARIS_ROOT", "polaris"))
sys.path.insert(0, str(POLARIS / "scripts"))

import isaaclab.app as _app  # noqa: E402  (importing the module does not launch Isaac)
from isaaclab.app import AppLauncher as _Launcher  # noqa: E402  (only a from-import resolves it)


#: Seconds an evaluation episode may run. PolaRiS's environments default to 30 s, which is
#: 450 steps at 15 Hz, and that is too short for this task: 38% of the episodes in the
#: training set are longer than 450 steps (median 348, p90 665, max 969), because episodes
#: that would have broken the FR3 motion budget are stretched in time rather than dropped.
#: Left at 30 s the evaluation truncates more than a third of the task instances before the
#: pour can happen, and reports it as failure - a number about the horizon, not the policy.
#: 70 s covers the longest episode in the data with room to spare.
EVAL_HORIZON_S = float(os.environ.get("POUR_EVAL_HORIZON_S", 70.0))


class _RegisterOurs(_Launcher):
    """PolaRiS's AppLauncher, with our environments registered as soon as Isaac is up.

    The horizon is patched in here too, for the same reason the registration is: their
    script reads the environment config immediately after launching Isaac, and this is the
    only point between the two.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import r2s2r.environments  # noqa: F401  (its import runs gym.register)

        import isaaclab_tasks.utils as _tasks

        _parse = _tasks.parse_env_cfg

        def parse_env_cfg(*a, **k):
            cfg = _parse(*a, **k)
            cfg.episode_length_s = EVAL_HORIZON_S
            print(f"[eval] episode horizon {EVAL_HORIZON_S:.0f} s "
                  f"({int(EVAL_HORIZON_S * 15)} steps at 15 Hz)")
            return cfg

        _tasks.parse_env_cfg = parse_env_cfg


_app.AppLauncher = _RegisterOurs

import tyro  # noqa: E402
from polaris.config import EvalArgs  # noqa: E402


def main():
    args = tyro.cli(EvalArgs)
    import eval as polaris_eval  # noqa: E402  (binds the patched AppLauncher)

    polaris_eval.main(args)


if __name__ == "__main__":
    main()
