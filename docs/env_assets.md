# How PolaRiS environment assets are organised

What the official PolaRiS-Hub environments look like, and the conclusions that still apply to
ours. The scene splat itself comes from GSWorld (see `docs/gsworld_to_polaris.md`); this page
is about the layout the environment code expects.

Based on the six official environments in `owhan/PolaRiS-Hub` plus the compositing code in
`src/polaris/environments/manager_based_rl_splat_environment.py`.

## Directory layout

An environment is one directory with three things:

```
food_bussing/
  scene.usda
  initial_conditions.json
  assets/
    g60_corner_charuco_static/   config.yaml  mesh.usdz  splat.ply  splat_clip_loss.ply
    battery/                     mesh.usdz
    blue_cup/                    mesh.usdz
    grapes/                      mesh.usdz
    ice_cream/                   mesh.usdz
    yellow_bowl/                 mesh.usdz
```

`scene.usda` has, under `/World`, a list of prims with `PhysicsRigidBodyAPI` and one Camera
called `external_cam`; each prim payloads its own `assets/<name>/mesh.usdz`. `droid_cfg.py`'s
`dynamic_setup` walks the children of `/World`: a Camera becomes a `CameraCfg` at its pose, a
rigid body becomes a `RigidObjectCfg`. Which objects exist and where the camera sits is
decided entirely by this file, not by code.

`initial_conditions.json` is `{"instruction": ..., "poses": [{object: [x, y, z, qw, qx, qy, qz]}, ...]}`;
one pose set is one rollout's initial layout.

## Scene from splats, objects from meshes, robot from per-link splats

The key fact: in all six official environments only the scene asset has a `splat.ply`; the
manipulated objects have only `mesh.usdz`. Raytracing the objects is the official practice,
not a fallback.

Compositing (`custom_render`) renders the splat image first, then pastes over it the pixels
whose semantic id ≥ 2 in Isaac Sim's raytraced render. Labels are assigned in
`setup_splat_world_and_robot_views`: any rigid object without `assets/<name>/splat.ply` gets
`class=raytraced`, so "splat if available, raytraced otherwise" is automatic. We tag the
`Gripper` subtree with the same label so the gripper is raytraced
(`src/polaris_lfhv/environments/__init__.py`).

The robot is the third kind: `nvidia_droid/SEGMENTED/` holds one ply per link, transformed by
link pose at run time and rendered together. `droid_cfg.py`'s `robot_splat` flag set to False
raytraces the whole robot instead.

The official scene splats contain no robot: rendering `food_bussing`'s scene splat alone shows
only the mounting plate and bolts, so the arm was removed or moved out of frame during their
capture. Our cell cannot do that (removing the arm changes the base pose and invalidates the
extrinsics and the demonstration), so the arm has to be separated afterwards; GSWorld's
per-gaussian link labels are what makes that possible.

## Why the tabletop gets a separate collision slab

The fused scene mesh is unusable as a tabletop collider. Measured: casting rays down over the
placement area, 621 samples hit the mesh only 506 times (holes elsewhere) with a 51 mm height
spread, and at the bottle's position the surface sits 10 mm above the placement height, so a
spawned object starts inside the table and PhysX ejects it. A low-texture black cloth is hard
for TSDF fusion.

PolaRiS has a precedent: `PolaRiS-Hub/pan_clean/scene.usda` hangs a Mesh called `platform`
under the scene prim, a scaled unit cube, convexHull, invisible. We do the same with a
1.10 × 1.20 m slab, 0.1 m thick, top face at the table height (-0.020), and clip the fused
mesh at -0.10 so its uneven tabletop takes no part in collisions.

The slab size is chosen, not measured: x -0.30..0.80, y -0.60..0.60 covers the FR3's reach;
the placement range (x 0.36..0.60, y ±0.24) is well inside it.

## Real-vs-sim material

`$GSWORLD_ROOT/data/random_rollouts_20260710/20260710/` holds eight real recordings:
home_static, randomwalk_1..4, wristroll_1..2, vertical_1. Each has the two DROID cameras
(third-person ZED 2i, wrist ZED Mini) as HD720 rectified-left RGB plus uint16 millimetre
depth, `cam_K.txt`, `timestamps.jsonl` and `robot_state.jsonl` (15 Hz qpos and ee on the same
wall clock); 8720 states in total. `20260709_camcheck/` is a 120-frame static segment at the
home pose. These are physical measurements and are used as such; they are not redistributed.

Pairing (as GSWorld documents, re-checked): nearest-neighbour match of frame timestamps to
`robot_state.jsonl`, cropped to the state log's window; worst gap 33 ms. `fr3_link8` is the
Polymetis end effector. The cameras were not moved between the 07-07 calibration and the
07-10 recordings (sub-pixel ECC shift on the camcheck segment).

One lesson kept: a single-marker calibration of the external camera is weak along the
optical axis (93 mm error measured once), so a fixed camera's pose must come from
registration against the reconstruction or from several markers, never from one marker's
PnP. The current camera pose is solved from depth against the FK'd arm
(`docs/gsworld_to_polaris.md`).

## Cables (GSWorld's practice, verified)

- Table-top cables stay. Deleting a real static object exposes unobserved space that has to
  be painted over and produces translucent floaters (their v5 decision).
- Their final removes two things: the two ArUco markers on the table and the cables hanging
  from the arm and gripper (pruned by URDF distance plus a brightness threshold). Both of our
  asset variants inherit this cleanup; the hanging cables leave with the robot split.

## Scale of the official scene meshes

Measured with a USD BBoxCache: lab_bench 2.65×2.40×1.38 m, 1.29 M faces; g60_stovetop_zed
1.85×0.76×1.35 m, 1.00 M faces; g60_corner_charuco_static 2.39×1.52×2.15 m, 78 k faces. The
official practice is to cut about two metres around the base and a metre or so high, at
0.1-1 M faces. The scene's `config.yaml` is IsaacLab's MeshConverter record: source
`untextured.glb` (bare mesh, appearance comes from the splat),
`collision_approximation: meshSimplification`, `mass: 1.0`. The scene mesh is for collisions
only.

## Cameras

The external camera is in `scene.usda`; `dynamic_setup` reads its translate and orient into a
`CameraCfg` at a fixed 720×1280. The official usda's Camera prim references a `camera.usdz`
from the author's machine, which produces a harmless USD warning. The wrist camera is not in
the usda; it is hard-coded in `droid_cfg.py`'s SceneCfg under
`robot/Gripper/Robotiq_2F_85/base_link/wrist_cam`. Our wrist camera's mount, field of view
and principal-point offset are overridden in `PourMustardCfg`; the numbers and their sources
are in its docstring.
