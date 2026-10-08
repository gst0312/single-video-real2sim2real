# phase5_eval

Simulation evaluation of the fine-tuned policy on held-out placements. The 30 conditions in
`eval_conditions.json` come from the same sampler as the training `initial_conditions.json`
with a different seed and the same range (bottle x 0.402-0.490, y -0.147..-0.048; cup x
0.442-0.499, y 0.128-0.181), so this measures same-distribution generalisation, not
extrapolation. Scripts: `scripts/eval/`; alignment of conventions: `docs/real_robot.md`.

Settings, identical for every checkpoint: 30 conditions, 70 s per episode (1050 steps),
open-loop horizon 8. 70 s is not the default; PolaRiS's 30 s (450 steps) would truncate the
38% of training-length episodes longer than 450 steps before the pour.

| checkpoint | success | rate | mean progress | file |
|---|---|---|---|---|
| base (no fine-tuning) | 0/3 | 0% | 0.667 | `base_zero_shot.csv` |
| 1000 steps | 2/5 | 40% | 0.800 | `sweep/step1000.csv` |
| 2000 steps | 17/30 | 57% | 0.833 | `sweep/step2000.csv` |
| 5000 steps | 29/30 | 97% | 0.989 | `sweep/step5000.csv` |
| 7000 steps | 24/30 | 80% | 0.922 | `sweep/step7000.csv` |
| 8000 steps | 28/30 | 93% | 0.978 | `sweep/step8000.csv` |
| 9999 (final) | 30/30 | 100% | 1.000 | `sweep/step9999.csv` |

Base and 1000-step rows were run on a few conditions to validate the chain; from 2000 steps
every row is the full 30. The final checkpoint is saved as 9999 because openpi counts from 0.

The 7000-step 80% is sampling variation on a plateau, not a regression: 5000 is 97%, 8000
93%, final 100%. From 5000 steps on, single evaluations fluctuate between 93 and 100%.

Progress is the number of rubric criteria reached (reach, lift, pour) over three, so 0.667
means "grasped but did not pour". The base policy scores 0.667 on all three episodes and
runs the full 1050 steps each time: it already grasps and lifts; fine-tuning adds the pour.

Failure structure at 2000 steps: of 13 failures, 10 grasped without pouring, 3 did not
complete the grasp; success and failure groups have near-identical object positions (means
within 1 cm) and carry distances (1.0 cm apart), so failures do not cluster in the workspace.
Recompute with `python scripts/eval/analyse_eval.py <csv> <conditions.json>`.

Videos (base episodes, three final-checkpoint rollouts at 8x and re-rendered at real speed)
are listed in `../README.md` for the Release.
