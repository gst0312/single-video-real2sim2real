# Real-robot session, 2026-08-14: 13 episodes, 4 successes, the last three consecutive

The first time a simulation-trained checkpoint ran on the real FR3. Zero-shot: nothing added
at evaluation time, conventions as in `docs/real_robot.md`. Episodes ep09, ep11, ep12 and
ep13 completed the pick, carry and pour; ep09 was the first success, ep10 failed again
(grasped without lifting), and after the placement was moved closer ep11-ep13 succeeded in a
row.

Launcher `scripts/real_robot/run_eval_safe.py`, instruction `pour the mustard into the blue
cup`, checkpoint 9999 (10k steps), open-loop horizon 8, 1050 steps (70 s) per episode.

`episodes.csv` indexes the episodes (recording and run timestamps, raw score). The videos
are not in the repository (compressed copies 0.2-0.5 MB each, originals 22-31 MB each;
`results/README.md`). `first_frames/` holds the dual-view first frame of the run that produced
ep09 and of the run that produced ep10-ep13, used to check placement.

| episode | time | steps | raw score | outcome |
|---|---|---|---|---|
| ep01 | 01:21:43 | 1049 | 0 | failed |
| ep02 | 01:34:11 | 1049 | 0 | failed |
| ep03 | 01:37:34 | 1049 | not recorded | scoring prompt was skipped (`float('')` ended the process); video kept |
| ep04 | 01:48:06 | 1049 | 0 | failed |
| ep05 | 01:55:47 | 1049 | 0 | failed |
| ep06 | 01:58:27 | 1049 | 0 | failed |
| ep07 | 02:00:24 | 1049 | 0 | grasped the bottle, did not lift (see below) |
| ep08 | 02:06:16 | 680 | 0 | stopped early |
| ep09 | 02:08:21 | 1049 | 0.01 | **success**, recorded with `--record-wrist` |
| ep10 | 02:12:36 | 1049 | 0 | grasped, did not lift (same symptom as ep07) |
| ep11 | 02:15:27 | 1049 | 0.01 | **success** |
| ep12 | 02:18:06 | 1049 | 0.01 | **success** |
| ep13 | 02:20:58 | 1049 | 0.01 | **success** |

The 0.01 on successes is a bug in the upstream scoring script (`y` assigned 1.0 and divided
by 100 with the numeric path), fixed in the launcher afterwards; the raw CSVs keep the
original value. The outcome column is the human-verified one.

## What the session established

The execution layer was fine: with `--trace` the follow error stayed at 0.01-0.06 rad and the
command step at 0.2-1.1 rad/s throughout the failures, so neither the controller nor the
policy stalled.

Gripper timing and scale lined up once the "hold until the gripper reads closed" patch was
in: the reading settled at 0.339-0.344 on the held bottle against 0.376 in simulation, and the
fingers stopped on the bottle after 5 steps rather than closing fully (which takes 28 steps
and reads near 1.0).

Left and right were correct: the demonstration and its reconstruction both have the bottle on
the left and the cup on the right, as on the table.

The failures shared one feature: the bottle was too far away. In the wrist view at the home
pose the bottle fills the lower half of the frame in a successful sim episode, almost at the
fingertips; in the failed real episodes it was a small patch in the upper middle. In ep07's
exterior video the arm reaches the bottle around step 210, has knocked it over by step 230,
and then barely moves for 800 steps: the end effector arrived offset from the bottle, pushed
it over, grasped a lying bottle, and every observation after that was outside anything in
the training set. Beyond the x/y boxes in `docs/real_robot.md`, the practical test is
therefore the wrist view: the bottle's size and position at the start should match a
successful sim episode's first frame, which `scripts/real_robot/check_placement.py` measures.

## Not yet established

4/13 is not a success rate: nine of the failures came before the placement issue was
recognised. The informative part is the end of the session: after the placement was moved
in, three consecutive successes plus the earlier ep09, which says that placing inside the
distribution is the deciding variable. A real success rate needs ten or more episodes at a
fixed placement without mid-session adjustment, which has not been run. The execution-layer
limits on the robot's controller were also not re-read before this session; the gate still
runs on the transcribed values.
