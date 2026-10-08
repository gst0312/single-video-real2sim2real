# Trajectory synthesis: first-round gate table

The synthesiser's first full run over the 50 initial conditions, 8 attempts each
(`scripts/datagen/synthesize_trajectories.py`), with the gates as they stood before the
grasp pre-screen and the time rescaling were added: 108 of 400 attempts passed every gate
(limits, velocity, IK error), covering 44 of 50 conditions; the dominant failure was the
velocity gate. Each episode is about 270 steps at 15 Hz including the home ramp; actions are
seven absolute joint positions plus a binary gripper.

After the changes recorded in `docs/design.md` (grasp-shape screen, key-pose IK pre-screen
against the envelope's zero-speed bands, re-synthesis at a slower time scale instead of
rejection) the per-condition pass rate rose to about 39%; the production batches that fed
the dataset (216 trajectories through the kinematic gate, 182 through the physics gate) were
run with the final gates and are summarised in `docs/results.md`. This table is kept as the
only per-condition record saved from the project.

| condition | passed / tried | passed limits | passed velocity |
|---|---|---|---|
| 0 | 1/8 | 3 | 4 |
| 1 | 0/8 | 0 | 6 |
| 2 | 2/8 | 2 | 7 |
| 3 | 2/8 | 2 | 6 |
| 4 | 2/8 | 2 | 6 |
| 5 | 3/8 | 3 | 4 |
| 6 | 3/8 | 3 | 7 |
| 7 | 0/8 | 0 | 5 |
| 8 | 1/8 | 1 | 4 |
| 9 | 3/8 | 3 | 4 |
| 10 | 2/8 | 2 | 6 |
| 11 | 3/8 | 3 | 3 |
| 12 | 4/8 | 5 | 7 |
| 13 | 2/8 | 2 | 4 |
| 14 | 2/8 | 2 | 7 |
| 15 | 2/8 | 2 | 4 |
| 16 | 3/8 | 3 | 6 |
| 17 | 2/8 | 2 | 4 |
| 18 | 0/8 | 1 | 4 |
| 19 | 0/8 | 0 | 6 |
| 20 | 3/8 | 3 | 6 |
| 21 | 4/8 | 4 | 6 |
| 22 | 3/8 | 3 | 6 |
| 23 | 0/8 | 1 | 1 |
| 24 | 2/8 | 2 | 6 |
| 25 | 3/8 | 3 | 8 |
| 26 | 2/8 | 2 | 4 |
| 27 | 3/8 | 3 | 8 |
| 28 | 1/8 | 3 | 2 |
| 29 | 2/8 | 4 | 5 |
| 30 | 1/8 | 1 | 5 |
| 31 | 2/8 | 2 | 4 |
| 32 | 1/8 | 2 | 5 |
| 33 | 2/8 | 3 | 3 |
| 34 | 2/8 | 2 | 5 |
| 35 | 1/8 | 1 | 1 |
| 36 | 1/8 | 3 | 2 |
| 37 | 2/8 | 2 | 6 |
| 38 | 3/8 | 3 | 5 |
| 39 | 2/8 | 2 | 3 |
| 40 | 4/8 | 4 | 5 |
| 41 | 2/8 | 2 | 4 |
| 42 | 1/8 | 1 | 4 |
| 43 | 3/8 | 3 | 7 |
| 44 | 2/8 | 2 | 3 |
| 45 | 1/8 | 2 | 3 |
| 46 | 2/8 | 2 | 5 |
| 47 | 2/8 | 2 | 7 |
| 48 | 0/8 | 1 | 6 |
| 49 | 4/8 | 4 | 4 |
