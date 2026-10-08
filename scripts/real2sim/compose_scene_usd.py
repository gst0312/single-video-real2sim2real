"""Write the scene.usda that PolaRiS loads for our pour-mustard environment.

PolaRiS composes environments in a web GUI. We generate the same file instead, because the
marker alignment already gives exact numbers and a generated file is reproducible. The
layout copies PolaRiS-Hub/food_bussing/scene.usda: every prim under /World is either a
camera or a rigid body, which is exactly what droid_cfg.SceneCfg.dynamic_setup walks.

The scene asset is a kinematic body (it never moves); the objects are dynamic bodies whose
poses get overwritten per rollout from initial_conditions.json. Everything is already in
metric robot-base coordinates, so all transforms here are identity except the camera.

The external camera is our calibrated ZED 2i. Calibration gives camera-to-base in the
OpenCV convention (+z forward, +y down); USD cameras look down -z with +y up, so the
rotation gets flipped about x. Its focal length and apertures are set so that the USD
pinhole reproduces the measured fx and fy at 1280x720.
"""

import argparse
import json

import numpy as np

TEMPLATE_HEADER = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
"""

SCENE_PRIM = """    def "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend payload = @./assets/{name}/mesh.usd@
    )
    {{
        bool physics:kinematicEnabled = 1
        bool physics:rigidBodyEnabled = 1
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

{platform}    }}

"""

# The table top the objects actually rest on. Copied from the "platform" prim in
# PolaRiS-Hub/pan_clean/scene.usda: a unit cube, convex hull collision, invisible, scaled
# and placed by hand. The official environments use it for exactly this reason — the fused
# scene mesh is a reconstruction and its support surface is not flat enough to stand a
# bottle on. Ours varies by 5 cm across the placement area and has holes, because the table
# is black cloth and the arm masks removed supervision over the middle of it.
PLATFORM_PRIM = """        def Mesh "platform" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 6, 2, 3, 7, 4, 5, 1, 0, 4, 0, 2, 6, 5, 7, 3, 1]
            uniform token physics:approximation = "convexHull"
            bool physics:collisionEnabled = 1
            point3f[] points = [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]
            uniform token subdivisionScheme = "none"
            token visibility = "invisible"
            quatd xformOp:orient = (1, 0, 0, 0)
            double3 xformOp:scale = ({sx}, {sy}, {sz})
            double3 xformOp:translate = ({cx}, {cy}, {cz})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}
"""

OBJECT_PRIM = """    def "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend payload = @./assets/{asset}/mesh.usd@
    )
    {{
        bool physics:kinematicEnabled = 0
        bool physics:rigidBodyEnabled = 1
        quatf xformOp:orient = ({qw}, {qx}, {qy}, {qz})
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

"""

CAMERA_PRIM = """    def Camera "{name}"
    {{
        float2 clippingRange = (0.01, 10000000)
        float focalLength = {focal}
        float focusDistance = 400
        float horizontalAperture = {haperture}
        float horizontalApertureOffset = {hoffset}
        float verticalAperture = {vaperture}
        float verticalApertureOffset = {voffset}
        quatd xformOp:orient = ({qw}, {qx}, {qy}, {qz})
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

"""

FOOTER = """}

def Xform "Environment"
{
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}
"""


def euler_xyz_to_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def matrix_to_quat_wxyz(rot):
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", required=True)
    p.add_argument("--camera-pose-npz", default=None,
                   help="use this camera pose instead of the calibration's. The 2026-07-07 "
                        "calibration solved the external camera from a single 100 mm marker, "
                        "which leaves the optical-axis direction weakly constrained; measured "
                        "against real frames it sits about 90 mm too far back, and 97 percent "
                        "of that error is along the axis. Registering the real frames into the "
                        "reconstruction pins the same camera from hundreds of features across "
                        "the whole scene instead. See docs/env_assets.md.")
    p.add_argument("--camera-key", default="36087771_left")
    p.add_argument("--intrinsics", required=True, help="zed_intrinsics json")
    p.add_argument("--scene-name", default="table_static")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--focal-length", type=float, default=1.0476, help="USD focal length; apertures follow from it")
    p.add_argument("--centred-principal-point", action="store_true",
                   help="write no aperture offset, i.e. pretend the principal point is centred")
    # Prim name and asset directory are kept identical: the splat renderer looks for
    # assets/<prim name>/splat.ply, so a mismatch would silently change how an object is
    # rendered. Our objects have no splat and are meant to go down the raytraced path,
    # like every manipulable object in the official environments.
    p.add_argument("--objects", nargs="*", default=["mustard=mustard", "blue_cup=blue_cup"],
                   help="prim=asset pairs; initial poses come from initial_conditions.json at reset")
    p.add_argument("--object-z", type=float, default=-0.027, help="default object height; per-rollout poses override it")
    p.add_argument("--table-z", type=float, default=-0.028,
                   help="top face of the collision platform, i.e. the table plane measured from the markers")
    p.add_argument("--platform-x", type=float, nargs=2, default=[-0.30, 0.80])
    p.add_argument("--platform-y", type=float, nargs=2, default=[-0.60, 0.60])
    p.add_argument("--platform-thickness", type=float, default=0.10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    calib = json.load(open(args.calib))
    pose = calib["third"][args.camera_key]["pose"]
    cam_pos = np.array(pose[0:3])
    rot_cv = euler_xyz_to_matrix(*pose[3:6])
    if args.camera_pose_npz:
        d = np.load(args.camera_pose_npz)
        moved = d["pos"] - cam_pos
        axis = rot_cv[:, 2]
        print(f"camera pose overridden: moved {np.linalg.norm(moved) * 1000:.0f} mm, "
              f"of which {abs(moved @ axis) / np.linalg.norm(moved) * 100:.0f}% along the optical axis")
        cam_pos, rot_cv = d["pos"], d["rot"]
    # OpenCV camera axes -> USD/OpenGL camera axes
    rot_usd = rot_cv @ np.diag([1.0, -1.0, -1.0])
    quat = matrix_to_quat_wxyz(rot_usd)

    intr = json.load(open(args.intrinsics))
    cam_entry = next(c for c in intr["cameras"] if c["serial"] == args.camera_key.split("_")[0])
    rect = cam_entry["rectified"]["left"]
    fx, fy = rect["fx"], rect["fy"]
    haperture = args.focal_length * args.width / fx
    vaperture = args.focal_length * args.height / fy
    # The rectified ZED's principal point is off centre (cx 638.02, cy 357.27 at 1280x720).
    # A USD camera says that with an aperture offset, in the same units as the aperture:
    # one pixel is focal/fx wide. The signs are the way they are because moving the aperture
    # window one way moves the rendered content the other, and because USD's image y points
    # up while the intrinsics' points down; both were fixed by measuring which way the wrist
    # view moved (that camera's cx is 41 px off centre, so the direction is unmistakable).
    hoffset = (args.width / 2 - rect["cx"]) * args.focal_length / fx
    voffset = (rect["cy"] - args.height / 2) * args.focal_length / fy
    if args.centred_principal_point:
        hoffset = voffset = 0.0

    print(f"external camera at {np.round(cam_pos, 4)} in base frame")
    print(f"  optical axis (USD -z): {np.round(-rot_usd[:, 2], 4)}")
    print(f"  fx {fx:.3f} fy {fy:.3f} -> horizontalAperture {haperture:.5f} verticalAperture {vaperture:.5f}")
    print(f"  cx {rect['cx']:.2f} cy {rect['cy']:.2f} -> apertureOffset {hoffset:.6f} {voffset:.6f}")

    sx = args.platform_x[1] - args.platform_x[0]
    sy = args.platform_y[1] - args.platform_y[0]
    sz = args.platform_thickness
    platform = PLATFORM_PRIM.format(
        sx=round(sx, 6), sy=round(sy, 6), sz=round(sz, 6),
        cx=round((args.platform_x[0] + args.platform_x[1]) / 2, 6),
        cy=round((args.platform_y[0] + args.platform_y[1]) / 2, 6),
        cz=round(args.table_z - sz / 2, 6),
    )
    print(f"collision platform: {sx:.3f} x {sy:.3f} m, top face at z = {args.table_z}")

    body = TEMPLATE_HEADER
    body += SCENE_PRIM.format(name=args.scene_name, platform=platform)
    for spec in args.objects:
        prim, asset = spec.split("=")
        body += OBJECT_PRIM.format(
            name=prim, asset=asset, qw=1.0, qx=0.0, qy=0.0, qz=0.0, x=0.0, y=0.0, z=args.object_z
        )
    body += CAMERA_PRIM.format(
        name="external_cam",
        focal=args.focal_length,
        haperture=round(haperture, 6),
        vaperture=round(vaperture, 6),
        hoffset=round(hoffset, 6),
        voffset=round(voffset, 6),
        qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3],
        x=cam_pos[0], y=cam_pos[1], z=cam_pos[2],
    )
    body += FOOTER

    with open(args.out, "w") as f:
        f.write(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
