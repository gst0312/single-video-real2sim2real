# datagen

From one human demonstration to training data. The order is the dependency order:

1. `measure_tcp.py` measures the Robotiq grip centre and closing axis from the nvidia_droid
   USD and writes `tcp.json` (Isaac).
2. `sample_grasps.py` samples antipodal grasps on the bottle mesh (jaxmp's official sampler,
   wrapped the way rsrd does it; synthesis venv).
3. `synthesize_trajectories.py` assembles R2R2R's state machine, jaxmp batched IK and trajgen
   resampling into an offline generator of reference joint trajectories, and gates them on
   the FR3-and-Panda limits and the motion budget. No Isaac dependency; runs in the synthesis
   venv. The gate's maths is `r2s2r.limits`.
4. `preview_episode.py` renders a synthesised episode kinematically (joints and object poses
   written in, no physics) to check the shape of the motion before spending physics time (Isaac).
5. `replay_reference_episode.py` replays a reference trajectory with physics: the arm is
   driven by the 8-d DROID action, the bottle moves only if gripped, the rubric scores the
   pour, and both camera views are rendered in the same pass. This is the physics gate and
   the source of the training observations (Isaac). `--episodes <dir> --shard i/n` runs a
   directory in one process per GPU.
6. `build_rlds.py` writes the episodes that pass the physics gate (`r2s2r.physics_gate`)
   as a DROID-RLDS dataset in the official co-training schema (RLDS venv).
7. `check_units.py` pushes the written dataset through openpi's own loader and prints the
   actions as the model sees them, to catch a unit or convention error before training
   (PolaRiS/openpi venv).
8. `analyse_dataset.py` measures what the dataset spans: layouts, grasps, joint ranges,
   end-effector footprint (synthesis venv).

`measure_motion_budget.py` is the evidence behind step 3's thresholds: it measures velocity,
acceleration and jerk of real teleoperated episodes at 15 Hz. Conclusion in its docstring:
the velocity gate can follow the specification, the acceleration gate cannot (libfranka's
10 rad/s² is a 1 kHz limit; real 15 Hz data exceeds it), so the acceleration ceiling is the
measured maximum. `score_against_reference.py` ranks synthesised episodes by how closely
their motion shape matches the reference recipe (approach angle, palm orientation, which
joints carry the pour); it is a ranking tool, not a gate.

Isaac-side scripts are run through `../polaris_env.sh`. Where each piece comes from and how
it deviates from upstream is in `docs/provenance.md`.
