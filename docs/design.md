# Design

How the two upstream systems are combined, what had to be written, and which decisions
shaped the data. This is the current design; the staged plan it grew from is not kept.

## The split

PolaRiS is an evaluation stack: an IsaacLab 2.3.0 environment with splat-rendered cameras,
an 8-d action (seven absolute joint positions plus a binary gripper) at 15 Hz, rubric
scoring, an evaluation script and openpi training configurations. It ships no scene-making
tools (its documentation points at external ones) and no data generation (its co-training
dataset is published as a result). Real2Render2Real is a single-video data method: object
reconstruction and segmentation, 4D tracking of the demonstration, grasp sampling, a
grasp-and-follow state machine with batched IK and trajectory augmentation, and domain
randomisation. Its data generation has no physics (objects are posed kinematically), renders
meshes rather than splats, and runs on Isaac Sim 4.5 / Python 3.10, which does not fit in a
venv with PolaRiS's IsaacLab 2.3 / Python 3.11.

So the pipeline is split where the dependencies split. Real2Render2Real's synthesis organs
produce reference joint trajectories offline, in pure JAX/numpy, with no simulator. PolaRiS's
environment then replays each reference trajectory with physics, the rubric judges the task,
a physics gate judges executability, and the observations rendered during that same replay
become the training data, labelled with the reference trajectory itself. The hand-off between
the two is one `.npz` per episode: joint trajectory, gripper command, the object trajectory
the synthesiser assumed, and the condition index.

Boundary decisions that keep the integration honest to its upstreams: the robot model is
PolaRiS's `nvidia_droid` untouched (drive gains, solver, reset pose, gripper subtree, binary
gripper, 0.5 threshold). The real FR3 enters only through the limits: joint positions in the
FR3-and-Panda intersection and the FR3 velocity envelope, enforced on the generation side
because the simulator does not enforce velocity (`docs/fr3_limits.md`). Data format is
DROID-RLDS in the official co-training schema (`docs/data_format.md`). Environment, renderer
and dependencies follow PolaRiS.

## What had to be written

1. The pour criterion. PolaRiS ships `reach`, `lift` and `is_within_xy`; pouring needs the
   bottle tilted over the cup's mouth, above the rim, held for a while. Written as a closure
   factory in PolaRiS's style, geometry read from the assets' bounding boxes at run time,
   thresholds measured on the demonstration (`polaris_lfhv/environments/rubrics.py`,
   `polaris_lfhv/pour_geometry.py`). `reach` is measured to the object's bounding-box centre
   rather than its mesh origin, because GSWorld's bottle mesh has its origin at the base and
   PolaRiS's 0.2 m threshold would otherwise never fire on a held bottle (measured minimum
   0.221 m).
2. The physics replay. PolaRiS has no data-generation script. The episode loop is
   `polaris/scripts/eval.py` with "ask the policy" replaced by "read the next reference
   row"; reset, step and rubric are PolaRiS's (`scripts/datagen/replay_reference_episode.py`).
3. The RLDS writer, following the DROID builder template and read back through openpi's
   loader (`scripts/datagen/build_rlds.py`, `check_units.py`).
4. The synthesiser's assembly: Real2Render2Real's state machine, IK controller and trajgen
   organs taken out of their simulator class into an offline script, plus the gates
   (`scripts/datagen/synthesize_trajectories.py`, `polaris_lfhv/limits.py`).

Everything else is a thin wrapper around an upstream script or a measurement tool; the
per-file account is `docs/provenance.md`.

## Decisions that shaped the data

- Layouts are perturbations of the demonstration's layout, not independent draws over the
  table: bottle ±5 cm and ±10° yaw, cup ±3 cm and ±15° yaw around their demonstrated poses.
  Independent draws put the cup on the wrong side of the bottle half the time, and the first
  physics rollout swept the bottle through the cup (8.2 cm displacement). The cup gets a small
  draw so the policy learns to look for it rather than memorise a place.
- Grasp shape is constrained to how the demonstration is poured from: side grasp (approach
  10-40° below horizontal) with the closing axis aligned with the demonstrated tilt axis
  (|cos| ≥ 0.9), so the pour is a wrist roll rather than an arm swing. A top grasp passes
  every limit and pours by swinging the arm through the cup.
- Grasp candidates are pre-screened at three key poses (grasp, mid-transport, pour) against
  the joint limits clipped to where the FR3 velocity envelope still allows motion; a pose in
  the envelope's zero-speed band cannot be reached at any speed. This raised the episode
  yield from 7% to 39% without changing any criterion.
- An episode that only exceeds the motion budget is re-synthesised with the whole state
  machine stretched in time by the measured overshoot (up to 4x), which is what the real arm
  would do; episodes that fail on limits or IK are dropped, since slowing down cannot fix them.
- The acceleration ceiling is the maximum measured on real executed trajectories (20.8
  rad/s² at 15 Hz), not libfranka's 10 rad/s², which is a 1 kHz quantity that real 15 Hz
  data exceeds.
- The home-to-pregrasp transit is an explicit smoothstep ramp inside half the motion budget;
  an open-loop replay has no controller to chase the first target with.
- The physics gate keeps episodes that grasped (lift ≥ 6 cm), carried (bottle within 5 cm of
  the reference) and did not touch the cup (≤ 2 cm). Pour accuracy is scored by the rubric
  but is not a gate criterion: the rigid-follow assumption does not model contact compliance
  or liquid.
- Gripper label is the binary command actually sent; one exterior camera written to both
  exterior fields; no filter file (`docs/data_format.md`).
- Training: LoRA rather than full fine-tuning, batch 32 rather than the official 128 (that
  value belongs to a DROID-scale co-training mix; openpi's default and the single-task
  references use 32), 10k steps, official learning-rate schedule and norm stats, wrist camera
  in (the pour ends by aiming the spout, when the third-person view is occluded by the arm).
- Evaluation on held-out layouts from the same sampler, 70 s horizon so that no episode is
  truncated before the pour, several checkpoints rather than one.

## Risks that materialised, and one that did not

- Whether PolaRiS's stock Robotiq can hold the bottle at all was the main task risk; the
  first physics replay lifted it 16.3 cm (demonstration 16.5) with 0.92° rms tracking, so no
  contact calibration was needed.
- The environment's default 30 s horizon would have failed 38% of the task instances before
  the pour; the horizon follows the data.
- The gripper stroke time on the real robot (1.7 s against an instantaneous binary joint in
  sim) was the one dynamics gap found on the robot, and is compensated in the launcher until
  the generator leaves time for it.
- Placement outside the narrow training footprint explained the real-robot failures; it is
  now checked from the wrist image before an episode starts.
