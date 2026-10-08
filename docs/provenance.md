# Provenance

Every file in this repository either uses an upstream implementation unchanged, follows an
upstream pattern with recorded deviations, or is new tooling with no counterpart. This page
is that account. Reference pointers: PolaRiS = github.com/arhanjain/polaris commit 129abc4;
its renderer branch = zubair-irshad/polaris `gsplat-3dgs-renderer` baseline 1e3a6d4;
r2r2r = github.com/uynitsuj/real2render2real commit 0ac79eb; openpi =
Physical-Intelligence/openpi, PolaRiS's submodule bd70b8f (the training configuration was
verified byte-identical to openpi main 15a9616 on 2026-08-13); GSWorld =
github.com/luccachiang/GSWorld at 96594c1.

## External material taken as published

- PolaRiS stack, PolaRiS-Hub assets (HuggingFace `owhan/PolaRiS-Hub`, 1.7 GB: six
  environments and the `nvidia_droid` robot), base policy and norm stats
  (`gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`, never recomputed),
  the official co-training dataset (`owhan/PolaRiS-datasets`, read for the schema).
- franka_description 02afaae (version 2.8.1) for FR3 and Panda limits; libfranka main
  `include/franka/rate_limiting.h` for the velocity envelope, acceleration and jerk constants.
- SplatSim (github.com/qureshinomaan/SplatSim, ea07395) for the spherical-harmonics rotation
  and the per-link KNN split idea; hbb1/2d-gaussian-splatting (335ad61) for ply readers,
  PSNR/SSIM and mesh post-processing; jaxmp (uynitsuj/jaxmp d1ab9c7), jaxls (chungmin99/jaxls
  21219e08), trajgen (uynitsuj/trajgen 6b19f56), rsrd (uynitsuj/rsrd cca2256).
- IsaacLab 2.3.0's `scripts/tools/convert_mesh.py`, copied verbatim as
  `scripts/real2sim/convert_mesh.py`.

## Used unchanged

- Environment class: PolaRiS `manager_based_rl_splat_environment.py`.
- Environment configuration: PolaRiS `droid_cfg.py`; only the wrist camera's mount, field of
  view and principal-point offset are overridden (`PourMustardCfg`).
- Robot: PolaRiS `robot_cfg.py` with the `nvidia_droid` USD and `SEGMENTED` link splats; the
  only change is narrowing the arm's joint limits to the FR3-and-Panda intersection at run
  time with IsaacLab's `write_joint_position_limit_to_sim`; the USD is untouched.
- Simulation evaluation: PolaRiS `scripts/eval.py` and `policy/droid_jointpos_client.py`.
- Training: openpi's `pi05_droid_jointpos_polaris`, copied with five recorded changes
  (below).

## Package modules

| file | counterpart | deviation / note |
|---|---|---|
| `environments/__init__.py` | PolaRiS `environments/__init__.py` registration pattern, `droid_cfg.EnvCfg` | wrist camera from the lab's ZED Mini intrinsics and GSWorld's `wrist2eef` mount; `Gripper` subtree tagged `class=raytraced` with the same mechanism PolaRiS uses for objects without a splat; FR3 limits applied at run time; optional wrist response gain; two environment ids (`DROID-PourMustard`, `DROID-PourMustardAruco`) |
| `environments/rubrics.py` | PolaRiS `rubrics/checkers.py` closure factories and `Rubric` | new `pouring` criterion (geometry from the assets' bounding boxes, thresholds from the demonstration); `reach_centre` measures to the bounding-box centre instead of the mesh origin; `PourRubric.reset` also resets stateful criteria, as the base class's docstring asks |
| `pour_geometry.py` | none | the pour criterion's maths, factored out of `rubrics.py` so it runs and is tested without Isaac |
| `limits.py` | none (constants from franka_description, libfranka, the deployment config) | the gate maths, factored out of `synthesize_trajectories.py` for the same reason |
| `physics_gate.py` | none | the gate criteria shared by `build_rlds.py` and `analyse_dataset.py` |
| `alignment.py` | Umeyama 1991; the maths of `colmap model_aligner` | used by the solvers |
| `robot_links.py` | SplatSim `scripts/articulated_robot_pipeline.py` | samples link meshes from the USD stage instead of PyBullet depth renders; link poses from the articulation, never from USD under Fabric |
| `training/pour_mustard_config.py` | openpi `training/misc/polaris_config.py` | five changes: our RLDS dataset, `rlds_data_dir`, 10k steps, LoRA (both model variants, a matching `freeze_filter`, `ema_decay=None`), batch 32; everything else untouched |

## scripts/datagen

| file | counterpart | deviation / note |
|---|---|---|
| `measure_tcp.py` | none | reads the fingertip pad meshes from the `nvidia_droid` USD in the open and closed state; grip centre (0, 0, 0.158) in the link8 frame, closing axis link8 +y |
| `sample_grasps.py` | jaxmp `AntipodalGrasps.from_sample_mesh`; wrapper copied from rsrd `GraspablePart.from_mesh` | rsrd is not imported (it pulls the DiG/nerfstudio stack); `max_width` 0.08 for the Robotiq 2F-85 instead of rsrd's Franka-hand 0.04/0.045; 60 grasps instead of 20 to cover every layout |
| `synthesize_trajectories.py` | r2r2r `franka_coffee_maker.py` (state machine, timing constants, `_calculate_target_poses`), jaxmp `solve_ik_batched` with their controller weights, rsrd `get_grasp_augs`, trajgen `traj_interp_batch` + `generate_uniform_control_points_batch` | offline, no simulator; initial poses from `initial_conditions.json` rather than their random starts; binary gripper; release off; smoothstep home ramp; gates = r2r2r's three checks + FR3-and-Panda limits + libfranka envelope capped by the deployment wall + measured acceleration ceiling + cup clearance (five gripper proxy points); grasp-shape screen and key-pose IK pre-screen; time rescaling instead of rejection; the whole demonstrated path is carried rigidly onto the condition's cup before r2r2r's `proportion=0.6` retarget |
| `preview_episode.py` | write-in mechanics of `compare_to_real_rollout.py` | kinematic preview, no physics |
| `replay_reference_episode.py` | PolaRiS `scripts/eval.py` episode loop | policy query replaced by the reference row; horizon set from the trajectory; per-episode report (tracking, lift, bottle-vs-reference gap, cup displacement, rubric trace); `--episodes --shard` runs many per process |
| `build_rlds.py` | the DROID RLDS builder template (`droid_dataset_builder/droid/droid.py`), schema from `owhan/PolaRiS-datasets` | adds `exterior_image_2_left` (same frame) and the three instruction fields openpi's reader needs; physics gate applied here; `get_metadata` overridden because this is a single-file builder |
| `check_units.py` | none | pushes the dataset through openpi's loader; found three issues: missing `dlimp`, the gate missing from the builder, and `DeltaActions` subtracting in place on read-only arrays |
| `analyse_dataset.py` | none | coverage of layouts, grasps, joints, end-effector footprint |
| `measure_motion_budget.py` | none | velocity / acceleration / jerk of real teleoperated episodes at 15 Hz; the evidence for the gate thresholds |
| `score_against_reference.py` | none | ranks episodes by motion shape against the reference recipe |

## scripts/eval

| file | counterpart | deviation / note |
|---|---|---|
| `eval_policy.py` | PolaRiS `scripts/eval.py` | the evaluation loop is untouched; wraps `AppLauncher` to register our environment after Isaac starts and to set the 70 s horizon |
| `shard_conditions.py`, `run_eval.sh` | none | one conditions file per simulator process; staggered start |
| `analyse_eval.py` | none | outcome by layout |
| `log_action_rates.py` | PolaRiS eval with `DroidJointPosClient.infer` wrapped | records commanded actions, reports implied velocities |
| `render_1x.py` | same wrapping | renders every step for a real-speed clip |
| `make_report.py` | none | report page from CSVs and clips |

## scripts/real_robot

| file | counterpart | deviation / note |
|---|---|---|
| `run_eval_safe.py` | DROID `scripts/evaluation/evaluate_openpi.py` | that script is never edited; its source is patched in memory (seven exactly-once patches, `docs/real_robot.md`) and executed; NUC gripper start-up race worked around; abort-only watchdog; all credentials and machine names from the environment |
| `check_placement.py` | none | bottle apparent size in the wrist view against the sim median |
| `compare_obs.py` | none | same joints, sim vs real images, through the served policy |
| `archive_rollouts.py` | none | re-encodes session recordings for browsing |

## scripts/real2sim

| file | counterpart | deviation / note |
|---|---|---|
| `build_scene_assets.sh` | none; strings the scripts below in the settled order | two variants of one asset |
| `gsworld_splat_to_2dgs.py` | SplatSim `transform_shs` (copied verbatim), 2DGS `sh_utils.eval_sh` for the self-check | `--keep-3d` keeps three scale columns for the gsplat backend; marker opacity restored by position from `fr3_scene06_clean.ply` |
| `relight_splat.py` | none | one gain/offset per channel, optionally within a world-space box |
| `split_robot_splat.py` | SplatSim KNN split | membership decided by GSWorld's per-gaussian labels, KNN only assigns the prim; height-dependent claim radius for wrist hardware on the links |
| `cull_wrist_camera_body.py` | same idea as `kill_near_camera_gaussians.py` | hides the wrist camera's own reconstruction from the link splats it rides on |
| `clean_marker_patches.py` | GSWorld `replace_regions` / `paint_out_gs.py` | in-place recolouring instead of donor cloth (donors render with the view-dependent look of their old position) |
| `kill_near_camera_gaussians.py`, `kill_overtable_floaters.py` | GSWorld `final_paintout.json` volume kills | view-cone and fog-box volumes measured on real frames; over-table kill without a brightness criterion, exempting link labels |
| `destain_table.py`, `add_table_face.py`, `insert_table_backing.py`, `tune_backing_colors.py`, `append_backing_rows.py` | none | photometric and geometric fixes to the tabletop, each fitted against the pixel-aligned real frame |
| `compose_scene_usd.py` | `PolaRiS-Hub/*/scene.usda`; `pan_clean`'s `platform` prim; PolaRiS's aperture-to-fov arithmetic inverted | collision slab at -0.020 |
| `make_initial_conditions.py` | `PolaRiS-Hub/*/initial_conditions.json`; r2r2r's and the lab's layout draw | perturbations of the demonstration layout; bottle yaw from the tracked pose |
| `convert_mesh.py`, `prepare_scene_mesh.py`, `set_collision_approximation.py` | IsaacLab `convert_mesh.py` verbatim; 2DGS `post_process_mesh`; none | the last fixes IsaacLab 2.3.0 writing an invalid `triangleMeshSimplification` token (the valid one, used by the official assets, is `meshSimplification`) |
| `solve_camera_from_depth.py` | depth-camera hand-eye calibration, trimmed ICP | camera pose in the kinematic frame from real depth against the FK'd FR3 model, rms 8.1 mm |
| `solve_scene_to_depth.py`, `solve_frame_correction.py` | rigid ICP (Umeyama) | scene splat against back-projected static depth; arm-to-arm correction as the initial value |
| `solve_wrist_mount.py`, `wrist_camera_offset.py`, `markers_in_base_from_wrist.py` | none | wrist mount solving and conversion to `CameraCfg.OffsetCfg`; the current mount is GSWorld's `wrist2eef`, and a "refinement" against marker positions solved through the same wrist chain was found circular and reverted |
| `measure_table_edges.py`, `measure_table_height.py` | none | edge-alignment acceptance numbers; pinhole back-projection plus trimmed plane fit |
| `measure_render_error.py`, `summarise_rollout_pack.py` | 2DGS `image_utils` / `loss_utils` for PSNR and SSIM | per-region values, brightness ratio and silhouette IoU added |
| `export_lerobot_qpos.py` | LeRobot v2.1 on-disk layout | exports measured qpos (and optionally the inline images) per episode |
| `replay_rollout_qpos.py` | PolaRiS `scripts/eval.py` loop | recorded joint angles as absolute-position actions; horizon raised past PolaRiS's 30 s |
| `compare_to_real_rollout.py`, `render_rollout_video.py`, `compare_to_demo_frame.py`, `make_four_column.py` | GSWorld's `real_vs_sim` frame and video comparisons | objects placed by back-projecting the demonstration frame onto the table plane; hidden objects via USD visibility (Fabric transforms do not affect the RTX render) |
| `render_splat_from_zed_gsplat.py`, `render_splat_from_zed_inria.py`, `quick_render_scene.py` | PolaRiS `SplatRenderer.render_raw` / the gsplat branch; GSWorld's vendored inria rasteriser | background black and principal-point crop as GSWorld renders (`cam_maniskill2gs`); the inria path is kept as a renderer cross-check (MAE 0.08 against gsplat) |
| `detect_speck.py` | none | acceptance tool for raytraced-object leaks in the wrist view |
| `check_environment.py` | PolaRiS README minimal example | reset, step, render both views |

Patches to the gsplat renderer branch (kept in the `POLARIS_3DGS` checkout, not here):
third-order SH loaded from `f_rest` and passed as `sh_degree=3`; `POLARIS_SPLAT_BG` and
`POLARIS_SPLAT_SOFTEN` switches in the environment. Patches to openpi (in the
`POLARIS_ROOT/third_party/openpi` checkout): `DeltaActions` copies before subtracting; the
registration of our config is wrapped in `try/except ImportError`.

## Inherited inputs that are not upstream software

- The demonstration recording (284 frames RGB-D, 1280×720, 15 fps) and its tracked object
  trajectory (`take4.npy`, FoundationPose). The track is a processing product; rsrd's
  official tracker needs a multi-view object scan that was not made, so the track is used and
  recorded as "to be replaced".
- Camera calibration (ZED intrinsics, external camera, wrist mount): physical facts,
  re-verified by pixel alignment; the external camera is now solved from depth.
- GSWorld's object meshes (mustard bottle, cup) and scene splat with labels: used with the
  conversion recorded in `docs/gsworld_to_polaris.md`; condition for switching to a
  self-trained scene is a sharpness match at the deployment camera.
- Object masses, estimated (no scale available): bottle 0.10 kg (a near-empty prop), cup
  0.20 kg; PolaRiS leaves object mass to the user.
- A private teleoperation dataset (LeRobot v2.1, 228 episodes, 15 Hz, measured qpos): used
  only to replay measured joint angles in the simulator and to measure the real motion
  budget; not redistributed.
- The lab robot's execution-layer limits (hard 2.075 / 2.51 rad/s, soft wall 1.575 / 2.01):
  transcribed, internally consistent, stricter than every official number; to be re-read on
  the robot.

## Deliberately not used

- Code and conclusions of the lab's earlier pipeline (its environment, controllers, actuator
  gains, task-specific gates, thresholds and measurements); only the idea of a physics replay
  as a binary gate was kept, re-implemented.
- r2r2r's LeRobot converters (delta actions; incompatible with PolaRiS's absolute joint
  positions in RLDS).
- GSWorld's prescaled COLMAP products; `colmap model_aligner` from wrist-camera positions
  (degenerate: the wrist frames, looking at a textureless cloth, do not register).
- A self-trained 2DGS reconstruction of the scene (two rounds did not match GSWorld's
  sharpness); its scripts were retired and are not in this repository.
