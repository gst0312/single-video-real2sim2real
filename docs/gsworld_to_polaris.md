# Scene asset: converting GSWorld's scene06 into a PolaRiS environment

The environment's scene is not reconstructed by this project. GSWorld
(github.com/luccachiang/GSWorld) already contains a finished 3D-Gaussian-splat
reconstruction of the same table and robot cell (`scene06`), together with per-gaussian
robot-link labels, the capture joint angles and a scene-to-robot transform. This document
is the recipe for turning that asset into a PolaRiS environment anchored to the robot's
kinematic base frame, and the acceptance numbers.

GSWorld carries no licence file, so none of its assets, meshes, calibrations or recordings
are redistributed here. The scripts take their location from `GSWORLD_ROOT`; without a
GSWorld checkout this part of the pipeline cannot be run.

## Route

The conversion is a rigid transform with spherical-harmonics rotation; the splat is kept as
true 3DGS and rendered with the `gsplat-3dgs-renderer` branch of PolaRiS
(zubair-irshad/polaris, baseline 1e3a6d4) instead of PolaRiS's stock 2DGS surfel renderer.
Two measurements drove this:

- PolaRiS's native renderer only uses the DC colour term at run time (`SplatRenderer.init_models`
  leaves `active_sh_degree` at 0; zeroing `f_rest` renders byte-identical images).
- GSWorld's splat is genuine 3DGS: the smallest axis falls on each of scale_0/1/2 about a
  third of the time, so dropping scale_2 would flatten 64.9% of the gaussians the wrong way,
  and flattening to zero-thickness discs makes the tabletop a field of streaks at the
  deployment camera's grazing angle.

The gsplat branch is drop-in (`Camera` class byte-identical, `render` keeps the `p_mat`
permutation). On top of it this project patches: third-order SH (GSWorld's own simulator
renders full SH; DC-only differs from full SH by 14% on the tabletop, which is also why the
SH rotation is required), background colour and principal-point offset, and the
`POLARIS_SPLAT_*` switches. Everything must run through `scripts/polaris_env.sh`, which
selects that backend; the stock renderer would flatten the splat.

## Source asset

- Splat: `$GSWORLD_ROOT/assets/fr3_robotiq_assets/fr3_final.ply` (md5 fe571d43, 619k
  gaussians, SH degree 3). Lineage, verified by diff: `fr3_final` is `fr3_scene06_clean`'s
  618,804 gaussians with only opacity changed (67,766 rows hidden: the two ArUco quads, the
  cables hanging from the arm, glow) plus 185 appended donor clones; positions, colours and
  SH untouched. Table-top cables are kept (GSWorld's v5 finding: deleting a real static
  object exposes unobserved space and looks worse). The marker windows can be restored from
  `fr3_scene06_clean.ply` by copying opacity back by position.
- Per-gaussian link labels: `fr3_final_semantics_gs.npy`, int64, -1 scene, 1..21 the link
  index of their ManiSkill articulation.
- Frame: `$GSWORLD_ROOT/configs/fr3_robotiq_final.json`, `sim2gs_arm_trans` (rigid, metres)
  and the capture joint angles `robot_scan_qpos`.

## Settled recipe

Everything is in `scripts/real2sim/build_scene_assets.sh`, two variants of one asset:

    scripts/real2sim/build_scene_assets.sh noaruco   # GSWorld's final, marker free -> PolaRiS-Hub/pour_mustard (primary)
    scripts/real2sim/build_scene_assets.sh aruco     # markers restored               -> pour_mustard_aruco (alternate)

Steps (parameters as in the script):

1. Convert (`gsworld_splat_to_2dgs.py --keep-3d --extra-transform scene_correction_fk3.npz
   --restore-markers-from fr3_scene06_clean.ply`): rigid transform into the kinematic base
   frame with the SH rotation copied from SplatSim (self-check colour error 8e-7); both
   variants restore the marker windows.
2. Relight to the deployment camera (one gain/offset over the whole splat), before the split
   so that the robot links carry the same exposure.
3. Split the robot into 19 per-link splats by GSWorld's labels (`split_robot_splat.py
   --semantics`, posed with their `robot_scan_qpos`), into `nvidia_droid/SEGMENTED`.
4. Marker-free variant only: recolour the marker gaussians to cloth in place
   (`clean_marker_patches.py --recolor`: positions unchanged, DC colour from k-nearest normal
   cloth, a single-scalar brightness jitter of 0.15σ, higher SH zeroed, sub-visible rows
   pushed to opacity -15). Moving cloth from elsewhere was tried and rejected: it renders
   with the view-dependent look of its old position.
5. Cut the reconstructed ZED camera body and tripod in front of the deployment camera
   (`kill_near_camera_gaussians.py --max-depth 0.4`; 0.6 starts eating the table edge) and a
   fog box of over-exposed floaters between the cone and the table's front edge.
6. Destain the tabletop (`destain_table.py`): the reconstruction bakes a low-frequency stain
   field into the cloth, measured against the pixel-aligned real frame and divided out.
7. Add the table's front face as geometry coloured from the real frame (`add_table_face.py`,
   x = 0.712) and cut the bright overhang drape beyond x = 0.72.
8. Append an opaque backing sheet under the semi-transparent cloth
   (`append_backing_rows.py`, colours fitted against a real frame).
9. Compose `scene.usda` (`compose_scene_usd.py --camera-pose-npz zed_pose_fk.npz --table-z
   -0.020`) and write `initial_conditions.json` (`make_initial_conditions.py --table-z -0.019`).

The marker windows were measured, not assumed: the paper sits 20 to 60 mm above the nominal
tabletop in GSWorld's frame, so the restore window is a 0.15 m radius, -20 to +70 mm slab;
above +70 mm what GSWorld hid is genuine floating glow and stays hidden. The wrist camera
mount is GSWorld's `wrist2eef` (3 mm from DROID's official mount position; the lab's own
hand-eye solution was 27 mm off and renders the gripper 1.6-1.9x too large).

## Acceptance (measured 2026-08-11/12)

- Strict replica of GSWorld's published render (no relight, gripper from splats, same
  recorded qpos): zero displacement on four segments, MAE 6.4, gradient correlation 0.92-0.94;
  geometrically the same image. Their render has a -8 px vertical offset against the real
  frame; ours has the same -8, inherited from their extrinsics, not introduced.
- Environment level (gsplat backend, raytraced gripper, wrist response gain) against eight
  real recordings: exterior PSNR 18.8-19.0, whole-frame IoU 0.91-0.92, arm IoU 0.74-0.76
  (after the later exposure midpoint and marker cleanup: arm IoU 0.81-0.84, see
  `results/replay_validation/`). A per-channel wrist gain (0.817/0.822/0.809,
  `PourMustardEnv.WRIST_RESPONSE`) raised held-out wrist PSNR 17.76 → 19.89.
- Raytraced gripper (`RAYTRACED_SUBTREES = ("Gripper",)`): at 8 cm the splat gripper is a
  smear while the raytraced one lands within a few pixels of the real one; raytracing the
  whole robot would drop exterior arm IoU from 0.764 to 0.704, so only the gripper subtree.
- With objects and physics: reset to an initial condition and run 60 steps; objects settle
  by 1 mm; throughput 3.5 steps/s with both cameras.
- Action convention: replaying measured teleoperation qpos as absolute joint-position actions
  gives rms 0.9-1.8 degrees, one control period of lag, so no actuator identification is
  needed. The environment horizon defaults to 30 s; longer replays need `episode_length_s`
  raised.

## Frames and camera (settled 2026-08-12, anchored to ground truth)

Everything is anchored to the kinematic base frame, the frame joint-position actions live in.
Three frames were found not to coincide: the kinematic frame, the marker frame (two-marker
fit, z normal to the table) and GSWorld's own anchor; the physical reason is that the
tabletop is tilted about 1.3-1.5 degrees against the robot's base plane (depth-fitted table
tilt 1.30 degrees under the FK camera; the two-marker normal (0.017, 0.023, 0.9996) ≈ 1.6
degrees is the same fact).

- External camera: `zed_pose_fk.npz`, solved directly from real ZED depth of the arm over
  eight poses against the FK'd FR3 model (`solve_camera_from_depth.py`, trimmed-ICP rms
  8.1 mm); 23.6 mm / 2.27 degrees away from GSWorld's extrinsics.
- Scene: `scene_correction_fk3.npz` = depth-ICP correction of the scene splat
  (`solve_scene_to_depth.py`, 1.52 degrees + 15 mm in z, composed on the arm-to-arm ICP)
  plus two small edge-probe deltas so that all four table edges land within ±0.3 px.
- Table height: platform top -0.020, objects spawn at -0.019 (cloth centre -0.0195 in the
  kinematic frame, plane rms 0.7 mm; the 1.3 degree cloth tilt leaves ±6 mm across the
  placement area, accepted for a flat platform).
- Acceptance (home pose, region misalignment before → after): far table edge -7 → 0 px, arm
  top -10 → -3, arm middle 0 → -1, neighbouring arm top -10 → -5; near table edge -1 → +6/-8,
  the floor of a rigid fit to a slightly warped reconstruction.
- Remaining small items: the front of the table renders 8 grey levels dark; table height
  should be re-checked on the robot by touching the table with the gripper.

Why the GSWorld asset is used instead of a self-trained reconstruction, and the condition for
switching back (a self-trained 2DGS matching GSWorld's sharpness at the deployment camera), is
recorded in `docs/provenance.md`.
