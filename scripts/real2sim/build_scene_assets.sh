#!/bin/bash
# From GSWorld's finished scene06 splat to a PolaRiS environment: the settled route
# (docs/gsworld_to_polaris.md, settled recipe):
#
#   convert   gsworld_splat_to_2dgs.py --keep-3d   rigid transform to the base frame with
#             the SH rotation, three scale columns kept for the gsplat backend
#   relight   one gain/offset onto the whole splat, fitted against the deployment camera,
#             before anything is split off (links split later must carry it too)
#   split     the robot leaves the scene by GSWorld's own per-gaussian labels, one ply per
#             prim into nvidia_droid/SEGMENTED (identical for both variants)
#   cut       the reconstructed ZED body and tripod in front of the deployment camera,
#             0.4 m deep (measured: 0.6 eats the table edge, 0.75 the cloth's backing),
#             plus the fog box between the cone and the table's front edge (overexposed
#             floaters)
#   destain   the cloth's baked low-frequency stain field, divided out against the
#             pixel-aligned real frame (destain_table.py)
#   face      the table's front face added as real-coloured geometry at x=0.712 and the
#             bright overhang drape beyond x=0.72 cut (the reconstruction has almost no
#             face gaussians; the drape painted the edge as a structureless bright band)
#   compose   scene.usda with the depth-solved camera pose and the platform at -0.020,
#             initial conditions at -0.019 (1 mm settle clearance)
#
# Two variants of the same asset, differing only in the two ArUco markers. The primary
# environment is the marker-free one, matching GSWorld's shipped final (2026-08-12 user
# decision); the marker variant is kept alongside. Both convert fr3_final WITH the marker
# gaussians restored from fr3_scene06_clean - the difference is what happens to them:
#   noaruco   markers recoloured to cloth in place (2026-08-13; GSWorld's own paint-out
#             donors, kept in fr3_final, read as table smudges once relit - moved cloth
#             carries the view-dependent look of its old position) -> PolaRiS-Hub/pour_mustard
#   aruco     markers kept                                         -> pour_mustard_aruco
# The dangling arm cables stay painted out (their final's call); table-top cables stay
# (their v5 call).
#
#   scripts/real2sim/build_scene_assets.sh noaruco
#   scripts/real2sim/build_scene_assets.sh aruco
#
# The environment must afterwards be run through scripts/polaris_env.sh (gsplat backend);
# the scene splat is true 3DGS and the stock surfel renderer would flatten it.
#
# Do NOT run this while renders are up: the split step clears and rewrites the SHARED
# nvidia_droid/SEGMENTED link splats, and an environment booting inside that window loads
# an empty robot (only the raytraced gripper shows).
#
# Frames (2026-08-12, settled the third time, this time against ground truth): everything
# is anchored to the KINEMATIC base frame - the frame joint-position actions live in.
#   camera   zed_pose_fk.npz: solved from real ZED depth of the arm across eight poses
#            against the FK'd FR3 URDF (solve_camera_from_depth.py, trimmed rms 8 mm).
#   scene    scene_correction_fk.npz: the converted splat ICP'd onto backprojected static
#            depth under that camera (solve_scene_to_depth.py), composed onto the
#            arm-to-arm correction. Neither the marker frame nor GSWorld's own anchor is
#            used: the physical tabletop (and with it the marker plane) is genuinely
#            tilted ~1.3-1.5 deg against the robot's base plane, so those frames disagree
#            with the kinematic one by construction.
#   table    the platform top is -0.020 and objects spawn at -0.019: the depth-measured
#            cloth height at the placement centre in the kinematic frame (-0.0195, plane
#            rms 0.7 mm; the 1.3 deg cloth tilt leaves +-6 mm across the placement area,
#            accepted for a flat platform).
set -e
VARIANT=${1:?usage: build_scene_assets.sh noaruco|aruco}
case "$VARIANT" in
    noaruco) ENV_NAME=pour_mustard ;;
    aruco)   ENV_NAME=pour_mustard_aruco ;;
    *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
POLARIS=${POLARIS_ROOT:?set POLARIS_ROOT to your PolaRiS checkout}
TWODGS=${TWODGS:?set TWODGS to the 2d-gaussian-splatting checkout}
W=${WORK:?set WORK to the working directory}
HUB=${HUB:-$POLARIS/PolaRiS-Hub}
GSWORLD_ROOT=${GSWORLD_ROOT:?set GSWORLD_ROOT to the GSWorld checkout (assets are not redistributed here)}
GSA=${GSA:-$GSWORLD_ROOT/assets/fr3_robotiq_assets}
GSC=${GSC:-$GSWORLD_ROOT/configs}
CAM=${CAM:-$GSWORLD_ROOT/real_robot_data/cameras}
B=$W/build_$VARIANT
ENV_DIR=$HUB/$ENV_NAME
PY=$POLARIS/.venv/bin/python
PY2=$TWODGS/.venv/bin/python
mkdir -p "$B" "$ENV_DIR/assets/table_static"

echo "=== [$VARIANT] convert to the base frame $(date -Is) ==="
# both variants restore the marker gaussians (and hide GSWorld's appended donor clones);
# the noaruco variant recolours the restored paper to cloth after the split
RESTORE=(--restore-markers-from "$GSA/fr3_scene06_clean.ply"
         --markers-json "$W/markers_in_base.json")
# scene_correction_fk3.npz: their frame -> kinematic frame (see the header); without it
# image residuals against real frames grow with the lever arm (10 px at the arm's top,
# 7 px at the far table edge). fk3 = fk composed with two edge-probe deltas solved
# 2026-08-13 against the real frame (per-sample 6-DoF, all four table edges land within
# +-0.3 px predicted). The neighbour corner (plate, arm) stays ~10 px high: that is
# reconstruction warp a rigid transform cannot also fix, tracked separately.
"$PY2" "$REPO/scripts/real2sim/gsworld_splat_to_2dgs.py" \
    --splat "$GSA/fr3_final.ply" --config "$GSC/fr3_robotiq_final.json" \
    --semantics "$GSA/fr3_final_semantics_gs.npy" --keep-3d "${RESTORE[@]}" \
    --extra-transform "$W/scene_correction_fk3.npz" \
    --out-prefix "$B/gs06" | tail -4

echo "=== [$VARIANT] relight to the deployment camera $(date -Is) ==="
"$PY" "$REPO/scripts/real2sim/relight_splat.py" --splat "$B/gs06_3dgs.ply" \
    --gain 1.2292 1.2763 1.2975 --offset 8.925 9.405 8.98 \
    --out "$B/gs06_relit.ply" | tail -1

echo "=== [$VARIANT] split the robot into per-link assets $(date -Is) ==="
# clear first: a link the split decides to skip would otherwise keep an older file and the
# robot would be drawn from two different runs. The shipped assets are in SEGMENTED_official.
rm -f "$HUB/nvidia_droid/SEGMENTED"/*.ply
"$REPO/scripts/polaris_env.sh" python "$REPO/scripts/real2sim/split_robot_splat.py" \
    --splat "$B/gs06_relit.ply" --qpos "$W/gsworld_scan_qpos.npy" \
    --semantics "$B/gs06_semantics.npy" \
    --out-dir "$HUB/nvidia_droid/SEGMENTED" \
    --scene-out "$B/gs06_scene.ply" --report "$B/split.json" \
    > "$B/split.log" 2>&1
[ -f "$B/split.json" ] || { echo "split produced no report, see $B/split.log"; exit 1; }
# the wrist camera must not photograph its own reconstruction (a dark dot pinned to the
# view centre); its body is hidden from the link splats it rides on
"$PY" "$REPO/scripts/real2sim/cull_wrist_camera_body.py" --segmented "$HUB/nvidia_droid/SEGMENTED"
"$PY" -c "
import json; d = json.load(open('$B/split.json'))
print('gaussians', d.get('gaussians'), 'on robot', d.get('on_robot'))"

echo "=== [$VARIANT] cut the reconstructed camera body $(date -Is) ==="
if [ "$VARIANT" = noaruco ]; then
    echo "=== [$VARIANT] recolour the markers to cloth $(date -Is) ==="
    # the markers leave by recolouring their own gaussians in place: the paper lies flat
    # on the cloth plane, so painting it beats every donor scheme - moved cloth renders
    # with the view-dependent look of its old position (see clean_marker_patches.py)
    "$PY" "$REPO/scripts/real2sim/clean_marker_patches.py" \
        --splat "$B/gs06_scene.ply" --semantics "$B/gs06_scene_semantics.npy" \
        --markers-json "$W/markers_in_base.json" --recolor \
        --out "$B/gs06_scene_painted.ply" --semantics-out "$B/gs06_scene_painted_semantics.npy"
    SCENE="$B/gs06_scene_painted"
else
    SCENE="$B/gs06_scene"
fi

"$PY" "$REPO/scripts/real2sim/kill_near_camera_gaussians.py" --splat "$SCENE.ply" \
    --pose "$W/zed_pose_fk.npz" --max-depth 0.4 \
    --fog-box 0.82 1.15 -0.8 0.8 -0.40 0.10 \
    --semantics "${SCENE}_semantics.npy" --out "$B/splat_cut.ply" | tail -3
cp "$B/splat_cut_semantics.npy" "$B/splat_semantics.npy"

echo "=== [$VARIANT] destain the tabletop against the real frame $(date -Is) ==="
# the reconstruction bakes a low-frequency stain field into the cloth (GSWorld's own
# render carries it too); measured against the pixel-aligned real frame and divided out
# per gaussian (destain_table.py). Fit view: home-adjacent real frame of the recordings.
DESTAIN_REAL=${DESTAIN_REAL:-$W/destain_real_frame_mid.png}
"$REPO/scripts/polaris_env.sh" python "$REPO/scripts/real2sim/render_splat_from_zed_gsplat.py" \
    --splat "$B/splat_cut.ply" --calib "$CAM/aruco_calib_result_20260707.json" \
    --intrinsics "$CAM/zed_intrinsics_live_20260707.json" \
    --pose-npz "$W/zed_pose_fk.npz" --polaris-3dgs "${POLARIS_3DGS:?set POLARIS_3DGS}" \
    --out "$B/destain_fit_view.png" > "$B/destain_render.log" 2>&1
"$PY" "$REPO/scripts/real2sim/destain_table.py" --splat "$B/splat_cut.ply" \
    --real "$DESTAIN_REAL" --sim "$B/destain_fit_view.png" --sigma 14 \
    --pose "$W/zed_pose_fk.npz" --out "$B/splat_destained.ply" | tail -1

echo "=== [$VARIANT] give the table its front face $(date -Is) ==="
# the true table edge is at x~0.70 (backprojected from the real frame; the platform's
# 0.80 footprint was a chosen approximation, not the measured table). The splat has
# almost no gaussians on the front face - those pixels were painted by a bright drape of
# overhang rows, which is why every photometric fix failed. The face is added as
# geometry coloured from the real frame (add_table_face.py) and the overhang drape
# beyond x 0.72 is cut so the curtain is what the edge rays hit.
"$PY" "$REPO/scripts/real2sim/add_table_face.py" --splat "$B/splat_destained.ply" \
    --real "$DESTAIN_REAL" --pose "$W/zed_pose_fk.npz" \
    --face-x 0.712 --z-range -0.22 -0.020 \
    --semantics "$B/splat_semantics.npy" --semantics-out "$B/splat_semantics.npy" \
    --out "$B/splat_faced.ply" | tail -1
mv "$B/splat_semantics.npy" "$B/splat_faced_semantics.npy"
# second box: 18 scene-labelled wrist-hardware crumbs hanging at the scan-pose wrist
# position (0.222, -0.138, 0.394) - the moving dark dot in wrist views; measured, the
# nearest other content is >15 cm away
"$PY" "$REPO/scripts/real2sim/kill_near_camera_gaussians.py" --splat "$B/splat_faced.ply" \
    --pose "$W/zed_pose_fk.npz" --max-depth 0.01 \
    --fog-box 0.72 1.15 -0.8 0.8 -0.20 0.04 \
    --fog-box 0.19 0.25 -0.17 -0.11 0.36 0.43 \
    --semantics "$B/splat_faced_semantics.npy" --out "$B/splat.ply" | tail -3

echo "=== [$VARIANT] append the fitted table backing $(date -Is) ==="
# the cloth is semi-transparent and dark junk under the table bleeds through as mottling;
# an opaque backing sheet under the cloth, geometry from insert_table_backing.py and
# colours fitted closed-loop against a real frame (tune_backing_colors.py), gives the
# translucency a clean backdrop. Stored rows re-append deterministically.
"$PY" "$REPO/scripts/real2sim/append_backing_rows.py" --splat "$B/splat.ply" \
    --semantics "$B/splat_semantics.npy" \
    --exposure 0.8429 0.8220 0.8135 -6.121 -6.057 -5.630 \
    --out "$B/splat_backed.ply" --semantics-out "$B/splat_backed_semantics.npy"

echo "=== [$VARIANT] compose the environment $(date -Is) ==="
"$PY" "$REPO/scripts/real2sim/compose_scene_usd.py" \
    --calib "$CAM/aruco_calib_result_20260707.json" \
    --intrinsics "$CAM/zed_intrinsics_live_20260707.json" \
    --camera-pose-npz "$W/zed_pose_fk.npz" \
    --table-z -0.020 --object-z -0.019 \
    --out "$ENV_DIR/scene.usda"
"$PY" "$REPO/scripts/real2sim/make_initial_conditions.py" --table-z -0.019 \
    --out "$ENV_DIR/initial_conditions.json"

echo "=== [$VARIANT] install $(date -Is) ==="
cp "$B/splat_backed.ply" "$ENV_DIR/assets/table_static/splat.ply"
cp "$B/splat_backed_semantics.npy" "$ENV_DIR/assets/table_static/splat_semantics.npy" 2>/dev/null || true
# collision mesh and object assets are variant independent; the primary environment owns
# them, the aruco variant fills in whatever it is missing from there
if [ "$ENV_NAME" != pour_mustard ]; then
    for piece in mesh.usd config.yaml; do
        if [ ! -e "$ENV_DIR/assets/table_static/$piece" ]; then
            cp "$HUB/pour_mustard/assets/table_static/$piece" "$ENV_DIR/assets/table_static/"
        fi
    done
    for obj in mustard blue_cup; do
        if [ ! -e "$ENV_DIR/assets/$obj" ]; then
            cp -r "$HUB/pour_mustard/assets/$obj" "$ENV_DIR/assets/$obj"
        fi
    done
fi

echo "=== [$VARIANT] smoke: reset, settle, render $(date -Is) ==="
case "$VARIANT" in
    noaruco) GYM_ID=DROID-PourMustard ;;
    aruco)   GYM_ID=DROID-PourMustardAruco ;;
esac
"$REPO/scripts/polaris_env.sh" python "$REPO/scripts/real2sim/check_environment.py" \
    --environment "$GYM_ID" --condition 0 --steps 10 \
    --out-prefix "$B/smoke" > "$B/smoke.log" 2>&1
[ -f "${B}/smoke_external_cam.png" ] || { echo "smoke render missing, see $B/smoke.log"; exit 1; }
echo "=== [$VARIANT] done: $ENV_DIR $(date -Is) ==="
