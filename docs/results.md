# Results

Every number in the README is listed here with the file it comes from. Files under
`results/` are the logs as written by the scripts; where a number exists only in the
project's generation or training logs (which are not in the repository), this page is the
record and says so.

## 1. Checkpoint sweep in simulation

Setting, identical for every checkpoint: 30 held-out placements (`eval_conditions.json`,
same sampler and range as the training layouts, seed 7: bottle x 0.402-0.490, y
-0.147..-0.048; cup x 0.442-0.499, y 0.128-0.181), 70 s per episode (1050 steps at 15 Hz;
PolaRiS's 30 s default would truncate 38% of the training-length episodes, 69 of 182, before
the pour), open-loop horizon 8, official norm stats. Scripts: `scripts/eval/`.

| checkpoint | success | rate | mean progress | source |
|---|---|---|---|---|
| base (no fine-tuning) | 0/3 | 0% | 0.667 | `results/phase5_eval/base_zero_shot.csv` |
| 1000 steps | 2/5 | 40% | 0.800 | `results/phase5_eval/sweep/step1000.csv` |
| 2000 steps | 17/30 | 57% | 0.833 | `results/phase5_eval/sweep/step2000.csv` |
| 5000 steps | 29/30 | 97% | 0.989 | `results/phase5_eval/sweep/step5000.csv` |
| 7000 steps | 24/30 | 80% | 0.922 | `results/phase5_eval/sweep/step7000.csv` |
| 8000 steps | 28/30 | 93% | 0.978 | `results/phase5_eval/sweep/step8000.csv` |
| 9999 (10k) steps | 30/30 | 100% | 1.000 | `results/phase5_eval/sweep/step9999.csv` |

Progress is the number of rubric criteria reached (reach, lift, pour) divided by three. The
base policy scores exactly 0.667 on all three of its episodes and runs the full 1050 steps:
it grasps and lifts but never pours, and time is not the limiting factor.

Failure structure at 2000 steps: of the 13 failures, 10 grasped but did not pour and 3 did
not complete the grasp (progress distribution in the CSV). Joining outcomes with layouts
(`scripts/eval/analyse_eval.py`): successes and failures have almost identical bottle and
cup position distributions (means within 1 cm, ranges fully overlapping) and carry
distances differing by 1.0 cm, i.e. failures do not cluster anywhere in the workspace, which
is why the remedy was more training rather than more data in one region.

The 7000-step 80% lies between 97% (5000), 93% (8000) and 100% (9999); a low point between
higher ones is evaluation variance on a plateau, not a regression. Evaluating several
checkpoints rather than one is what makes that visible.

Commanded speeds of the final checkpoint, two successful episodes, 2100 steps
(`scripts/eval/log_action_rates.py`): per-joint peaks 0.81 / 1.20 / 1.06 / 0.91 / 1.10 /
0.54 / 0.90 rad/s, i.e. 0.51 / 0.76 / 0.67 / 0.57 / 0.55 / 0.27 / 0.45 of the deployment wall,
2098 velocity samples, zero over-limit steps; gripper outputs exactly {0, 1}. Recorded in
`docs/real_robot.md`.

## 2. Dataset

| quantity | value | source |
|---|---|---|
| initial conditions sampled | 50 | `make_initial_conditions.py --count 50` (default) |
| trajectories through the kinematic gate | 216 (113 in the first batch + 103 in a second batch with seed 1) | generation log, 2026-08-13 |
| trajectories through the physics gate, written to RLDS | 182 | generation log, 2026-08-13; `build_rlds.py` prints the count |
| frames | 81,469 | training log, 2026-08-13 |
| RLDS size | 4.2 GB, 64 shards | generation log, 2026-08-13 |
| conditions represented | 43 of 50, median 4 episodes per condition | `analyse_dataset.py` run of 2026-08-13 |
| distinct grasps | 22 of 2880 candidates; the most used appears in 10% of episodes | same |
| joint coverage | j7 visits 97% of its allowed range | same |
| end-effector xy footprint | 57% of its bounding box, 2 cm grid | same; plot `results/figures/dataset_coverage.png` |
| episode length | median 348, p90 665, max 969 steps; 69 of 182 (38%) longer than 450 | same |
| bottle start box | x 0.403-0.501, y -0.144..-0.050 (10×9 cm), yaw ±10° | same |
| cup box | x 0.442-0.501, y 0.126-0.184 (6×6 cm), yaw ±15° | same |

The dataset itself, the checkpoint and the demonstration recording are not in the
repository.

First synthesis round, before the grasp pre-screen and time rescaling were added (27%
through the gate, 108 of 400 attempts, 44 of 50 conditions covered; failures mostly on the
velocity gate): per-condition table in `results/trajectory_synthesis.md`. After the changes
the per-condition pass rate rose to 39%; the production batches above were run with the
final gates.

Unit check (`scripts/datagen/check_units.py`, through openpi's own loader with the official
norm stats): after `DeltaActions` and normalisation the action means lie in -0.068..0.064,
standard deviations 0.152-0.292, all values within ±2.1; `image_masks` report the base and
left-wrist cameras as present. The official stats' action standard deviations are
0.094-0.171 on the seven arm dimensions and the gripper dimension has mean 0.451, std 0.441.

## 3. Training

| quantity | value | source |
|---|---|---|
| base checkpoint | `pi05_droid_jointpos_polaris` (openpi, PolaRiS's DROID joint-position model) | `pour_mustard_config.py` |
| method | LoRA (`gemma_2b_lora` / `gemma_300m_lora`, matching freeze filter, no EMA) | same |
| batch size | 32 (global), lr 5e-5 cosine, 1000 warm-up steps, action horizon 15 | same |
| steps | 10,000; checkpoints every 1000; final saved as 9999 | same |
| hardware | 4 × RTX 6000 Ada, data parallel (`fsdp_devices=1`), 1.9 s/it, about 5.3 h | training log, 2026-08-13 |
| passes over the data | 10k × 32 / 81,469 = 3.9 | arithmetic |
| loss | 0.032 at the start to 0.0028 at 5100 steps, still falling | training log, 2026-08-13 |
| norm stats | official DROID stats; the checkpoint's copy is byte-identical (md5 385156b4295b2efd7c342d5063bce321) | `docs/real_robot.md` |

## 4. Physics-gate smoke test (phase 3)

The first three synthesised trajectories replayed with physics, before the production
batch, all with time scale 2.0 and 568 steps (`results/phase3_rollouts/*.json`):

| episode | rubric | lift | tilt | mouth to cup axis | above rim | held | cup moved | tracking rms |
|---|---|---|---|---|---|---|---|---|
| cond000_ep06 | 3/3 | 15.7 cm | 82° | 0.8 cm | 6.0 cm | 76 steps | 0.000 m | 1.10° |
| cond000_ep05 | 3/3 | 15.4 cm | 79° | 1.0 cm | 6.6 cm | 74 steps | 0.000 m | 1.17° |
| cond000_ep03 | 3/3 | 14.6 cm | 71° | 1.5 cm | 8.1 cm | 67 steps | 0.000 m | 1.09° |

Cup radius 4.7 cm, so the mouth is inside the cup's opening. In the same batch 7 of 7
physics replays grasped the bottle, no cup moved, and the bottle stayed within 3.5 cm of its
reference trajectory. The demonstration's own pour, measured on its tracked trajectory: tilt
83-84°, mouth 1.9-2.2 cm from the cup axis, 6.3-7.0 cm above the rim, lift 16.5 cm.

## 5. Scene fidelity and action convention

Static frames against eight real recordings, three frames each, with the arm posed at the
recorded joint angles (`results/replay_validation/rollout_frames_summary.json`; per-segment
means):

| segment | ext PSNR | ext SSIM | ext IoU | wrist PSNR | wrist SSIM | arm IoU |
|---|---|---|---|---|---|---|
| home_static | 16.92 | 0.691 | 0.899 | 14.50 | 0.793 | 0.840 |
| randomwalk_1 | 16.99 | 0.692 | 0.897 | 13.98 | 0.784 | 0.837 |
| randomwalk_2 | 17.17 | 0.696 | 0.901 | 14.52 | 0.808 | 0.843 |
| randomwalk_3 | 16.60 | 0.684 | 0.890 | 14.35 | 0.774 | 0.814 |
| randomwalk_4 | 16.81 | 0.687 | 0.893 | 16.23 | 0.829 | 0.830 |
| vertical_1 | 16.39 | 0.679 | 0.887 | 15.43 | 0.813 | 0.815 |
| wristroll_1 | 17.04 | 0.693 | 0.900 | 15.52 | 0.817 | 0.841 |
| wristroll_2 | 16.37 | 0.679 | 0.886 | 15.48 | 0.816 | 0.817 |
| mean | 16.78 | 0.688 | 0.894 | 15.00 | 0.804 | 0.830 |

The global exposure was deliberately set halfway between GSWorld's native exposure and a fit
to the lab's July frames, which makes the simulator about 22% darker than those frames and
lowers PSNR against them relative to an exposure-matched render (exterior 19.0 → 16.8); the
IoU threshold was scaled with it. Joint write-in error is 0.0000° on every segment.

Replaying recorded joint angles as absolute joint-position actions (the tracking error of
PolaRiS's stock controller on this arm model), `results/replay_validation/`:

| source | episodes | rms (deg) | max (deg) |
|---|---|---|---|
| eight GSWorld recordings (random walks, wrist roll, vertical), 321-800 steps | 8 | 0.00-0.54 | 0.00-2.13 |
| teleoperation episodes (pick up the mustard; pour-type tasks), 159-788 steps | 8 | 0.93-1.81 | 4.10-9.01 |

rms is about the 95th-percentile commanded speed times one control period (1/15 s), i.e. a
one-step controller lag; no actuator identification was needed.

## 6. Real robot, 2026-08-14

First session, FR3, 10k checkpoint, zero-shot, instruction `pour the mustard into the blue
cup`, open-loop horizon 8, 1050 steps per episode (`results/real_robot_20260814/`):

| episode | time | steps | outcome |
|---|---|---|---|
| ep01-ep06 | 01:21-01:58 | 1049 each | failed |
| ep07 | 02:00 | 1049 | grasped the bottle, did not lift |
| ep08 | 02:06 | 680 | stopped early |
| ep09 | 02:08 | 1049 | success |
| ep10 | 02:12 | 1049 | grasped, did not lift (same symptom as ep07) |
| ep11, ep12, ep13 | 02:15-02:20 | 1049 each | success, three in a row after moving the placement closer |

Findings from the session: the execution layer tracked commands throughout (follow error
0.01-0.06 rad, command step 0.2-1.1 rad/s); the gripper reading on a held bottle was
0.339-0.344 against 0.376 in sim; failures shared one feature, the bottle placed too far
from the gripper (in the wrist view it covered 0.95-1.06% of the frame against 1.31% in the
median sim episode; every success measured 1.14-1.37%), which `check_placement.py` now
tests for. The "grasped, did not lift" symptom was the gripper's 1.7 s stroke (instantaneous
in sim); holding position until the gripper reads closed fixed it (`docs/real_robot.md`). The
scoring script recorded successes as 0.01 (fixed in the launcher; the archived CSVs keep the
raw value, the outcome column above is the human-verified one).

4/13 is not a success rate: nine of the failures preceded the placement finding. A
fixed-placement run of ten or more episodes has not been done, and the execution-layer
limits were not re-read on the robot before this session.
