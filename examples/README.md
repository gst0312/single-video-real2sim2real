# examples

One CPU-only walk-through of the pipeline's gating logic. No GPU, simulator, jax or external
checkout is needed; numpy is the only dependency.

```bash
python examples/gate_and_rubric_demo.py        # about 0.1 s
```

`data/sample_episode.json` is a synthetic 130-step episode at 15 Hz (joint trajectory, gripper
command, bottle and cup poses, object geometry) produced by `make_sample_episode.py`. It is an
illustration, not an IK solution of the real synthesiser, and its joint trajectory and object
track are not tied to each other. Real episodes come from
`scripts/datagen/synthesize_trajectories.py` (joints) and
`scripts/datagen/replay_reference_episode.py` (object poses from physics), both of which need
the stacks described in `docs/setup.md`.

What the script does, and the expected output:

1. Runs the kinematic gate (`polaris_lfhv.limits`) on the sample, on a copy played at double
   speed (fails the velocity budget at 1.12x), on that copy slowed by the synthesiser's rule
   (passes again), and on a copy that drives j6 below the FR3's 0.4398 rad floor (fails the
   limits; the velocity envelope's allowed speed there is zero, reported as "blocked", which
   is why slowing down cannot rescue such an episode).
2. Evaluates the pour criterion (`polaris_lfhv.pour_geometry`) step by step on the sample's
   object track with the 15-step dwell.
3. Applies the physics gate (`polaris_lfhv.physics_gate`) to the three real replay reports in
   `results/phase3_rollouts/` and to one synthetic failure.

```
sample episode: 130 steps at 15 Hz, gripper closes at step 36, instruction 'pour the mustard into the blue cup'

1. kinematic gate (FR3-and-Panda limits, velocity envelope, motion budget)
   velocity ceiling rad/s: [1.575 1.575 1.575 1.575 2.01  2.01  2.01 ]
   acceleration ceiling rad/s^2: 20.8
  as synthesised                130 steps  limits True  vel 0.56x ok   acc 0.10x ok   jerk 0.003x (reported)  -> PASS
  played at 2x speed             65 steps  limits True  vel 1.12x OVER  acc 0.36x ok   jerk 0.018x (reported)  -> FAIL
  2x copy slowed by 1.25x        81 steps  limits True  vel 0.90x ok   acc 0.24x ok   jerk 0.014x (reported)  -> PASS
  j6 driven to 0.40 rad         130 steps  limits False vel blocked OVER  acc 20.39x OVER  jerk 2.545x (reported)  -> FAIL
   velocity window at the final pose (rad/s): [-1.58 -1.58 -1.58 -1.58 -2.01 -2.01 -2.01] .. [1.58 1.58 1.58 1.58 2.01 2.01 2.01]

2. pour criterion on the sample's object track (tilt > 60 deg, mouth inside the cup's
   radius, mouth above the rim by 0 .. 0.30 m, held for 15 consecutive steps)
   step   0  tilt   0.0 deg  mouth-to-cup  25.6 cm  above rim   5.1 cm
   step  20  tilt   0.0 deg  mouth-to-cup  25.6 cm  above rim   5.1 cm
   step  40  tilt   0.0 deg  mouth-to-cup  25.6 cm  above rim   5.1 cm
   step  60  tilt   0.0 deg  mouth-to-cup  21.0 cm  above rim  13.1 cm
   step  80  tilt   0.0 deg  mouth-to-cup  16.6 cm  above rim  21.1 cm
   step 100  tilt  56.0 deg  mouth-to-cup   3.4 cm  above rim  13.8 cm
   step 102  tilt  61.6 deg  mouth-to-cup   2.8 cm  above rim  12.4 cm  over the cup
   step 116  tilt  84.0 deg  mouth-to-cup   2.0 cm  above rim   6.2 cm  over the cup
   step 120  tilt  84.0 deg  mouth-to-cup   2.0 cm  above rim   6.2 cm  over the cup
   criterion first met at step 102, success declared at step 116 (dwell 15); longest hold 28 steps

3. physics gate on the shipped phase-3 replay reports (lift >= 6 cm, drift <= 5 cm, cup <= 2 cm)
   cond000_ep03: lift 14.6 cm, drift 2.7 cm, cup moved 0.000 cm, rubric progress 1.00 -> kept
   cond000_ep05: lift 15.4 cm, drift 1.7 cm, cup moved 0.000 cm, rubric progress 1.00 -> kept
   cond000_ep06: lift 15.7 cm, drift 1.4 cm, cup moved 0.000 cm, rubric progress 1.00 -> kept
   synthetic failure: dropped because lift 1.2 cm; cup moved 8.2 cm
```

The unit tests in `tests/` cover the same modules with assertions
(`python -m pytest tests -q`). GPU, Isaac and real-robot steps are documented in `docs/` and
`scripts/*/README.md`; they are not bundled.
