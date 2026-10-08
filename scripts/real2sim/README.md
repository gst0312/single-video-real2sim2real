# real2sim

Turning the real scene and objects into a PolaRiS environment, and the tools that show it
matches the real robot.

The entry point is `build_scene_assets.sh`: it follows the settled recipe in
`docs/gsworld_to_polaris.md` from GSWorld's finished scene06 splat to an installed
`scene.usda`, one variant per invocation (`noaruco`, the primary environment, or `aruco`).
It needs a GSWorld checkout (`GSWORLD_ROOT`), whose assets are not redistributed here. The
other scripts fall into three groups:

- Build steps, called in order by the build script: conversion to the base frame with SH
  rotation (`gsworld_splat_to_2dgs.py`), relighting (`relight_splat.py`), splitting the robot
  into per-link splats by GSWorld's labels (`split_robot_splat.py`), hiding the wrist camera's
  own reconstruction (`cull_wrist_camera_body.py`), recolouring the marker patches
  (`clean_marker_patches.py`), cutting the reconstructed camera body and floaters
  (`kill_near_camera_gaussians.py`, `kill_overtable_floaters.py`), destaining the tabletop
  (`destain_table.py`), adding the table's front face (`add_table_face.py`), the opaque
  backing under the cloth (`insert_table_backing.py`, `tune_backing_colors.py`,
  `append_backing_rows.py`), composing the USD (`compose_scene_usd.py`), writing the initial
  conditions (`make_initial_conditions.py`), mesh conversion (`convert_mesh.py`,
  `prepare_scene_mesh.py`, `set_collision_approximation.py`).
- Solvers and measurements that produce the corrections the build consumes or the acceptance
  numbers: camera pose from depth against the FK'd arm (`solve_camera_from_depth.py`), scene
  against depth (`solve_scene_to_depth.py`), arm-to-arm correction (`solve_frame_correction.py`),
  wrist mount (`solve_wrist_mount.py`, `wrist_camera_offset.py`, `markers_in_base_from_wrist.py`),
  table edges and height (`measure_table_edges.py`, `measure_table_height.py`), render error
  (`measure_render_error.py`), teleoperation export (`export_lerobot_qpos.py`).
- Comparison and acceptance: renders (`quick_render_scene.py`, `render_splat_from_zed_gsplat.py`,
  `render_splat_from_zed_inria.py`), real-vs-sim videos and frames against the recordings
  (`compare_to_real_rollout.py`, `render_rollout_video.py`, `compare_to_demo_frame.py`,
  `make_four_column.py`, `summarise_rollout_pack.py`), speck detection (`detect_speck.py`),
  replay tracking error (`replay_rollout_qpos.py`), and the environment smoke test
  (`check_environment.py`).

Isaac-side scripts are run through `../polaris_env.sh`; the stock renderer would flatten the
3DGS splat. Each script's upstream counterpart and the reason for every deviation is recorded
in `docs/provenance.md`.
