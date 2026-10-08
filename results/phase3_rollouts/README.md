# phase3_rollouts

The first synthesised trajectories replayed with physics (2026-08-13), the check that had to
pass before any data was generated: can PolaRiS's stock Robotiq hold this bottle, and does
the pour avoid the cup?

Unlike the kinematic previews, where joints and object poses are written into the simulator
and the bottle follows no matter what, here the arm is driven only by the 8-d DROID action
(seven absolute joint positions plus a binary gripper); the bottle moves only if gripped and
the cup is a physical rigid body. The videos (not in the repository) show the exterior ZED 2i
view on the left and the wrist ZED Mini on the right, 15 Hz, 568 steps, time scale 2.0.

| episode | rubric | lift | tilt | mouth to cup axis | above rim | held | cup moved | tracking rms |
|---|---|---|---|---|---|---|---|---|
| cond000_ep06 | 3/3 | 15.7 cm | 82° | 0.8 cm | 6.0 cm | 76 steps | 0.000 m | 1.10° |
| cond000_ep05 | 3/3 | 15.4 cm | 79° | 1.0 cm | 6.6 cm | 74 steps | 0.000 m | 1.17° |
| cond000_ep03 | 3/3 | 14.6 cm | 71° | 1.5 cm | 8.1 cm | 67 steps | 0.000 m | 1.09° |

Cup radius 4.7 cm, so the mouth lands inside the opening. Of the same batch's seven replays,
7/7 grasped, no cup moved, and the bottle stayed within 3.5 cm of its reference.

The grasp shape follows the recipe: closing axis parallel to the demonstrated tilt axis,
approach from the side below horizontal, so the pour is a wrist roll rather than an arm
swing. Both objects were perturbed (bottle ±5 cm / ±10° yaw, cup ±3 cm / ±15° yaw) and the
pour target follows the cup.

Each `.json` is the complete report of one episode (rubric, tracking error, bottle-vs-
reference gap, cup displacement, pour-trace statistics), as written by
`scripts/datagen/replay_reference_episode.py`. The physics gate in `polaris_lfhv.physics_gate`
reads exactly these fields; `tests/test_physics_gate.py` and the example use these files.
