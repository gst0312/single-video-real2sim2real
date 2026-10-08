"""Turn 2DGS's unbounded mesh into a collision mesh for the scene asset.

The unbounded TSDF fusion reconstructs the whole room, which comes out at hundreds of
megabytes. The official PolaRiS scene assets are untextured meshes whose only job is
collision — appearance comes from the splat, and a prim that has a splat.ply is never
tagged raytraced, so this mesh is never drawn. Measured from the official assets, they
span roughly 2 m in x and y and 1.4 m in z and carry about a million faces, so crop to
a comparable volume and decimate to that order.

The default ceiling is just above the table plane on purpose. Our capture could not
remove the robot arms, so the fused mesh has phantom geometry wherever the training
masks left the model unconstrained; cutting at the table keeps the one surface that
matters for a tabletop task and admits none of it.

Crop bounds are in the robot base frame, which is what the reconstruction is already in.
Runs on open3d, the same library 2DGS uses to build and write this mesh; trimesh's ply
reader takes tens of minutes on a file this size.

The floater removal is 2DGS's own `utils.mesh_utils.post_process_mesh`, called rather than
reimplemented, so this step behaves exactly as it does inside their `render.py`. What is
ours is the crop box and the decimation target, both set from the official assets.
"""

import argparse
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def load_post_process_mesh(repo):
    """Import 2DGS's own floater filter, from its checkout."""
    repo = Path(repo)
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("mesh_utils", repo / "utils" / "mesh_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.post_process_mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True, help="output .obj")
    p.add_argument("--x-range", type=float, nargs=2, default=[-0.9, 1.3])
    p.add_argument("--y-range", type=float, nargs=2, default=[-1.1, 1.1])
    p.add_argument("--z-range", type=float, nargs=2, default=[-1.0, 0.02],
                   help="table plane sits at z=-0.03; the default ceiling clears it and nothing more")
    p.add_argument("--target-faces", type=int, default=1000000)
    p.add_argument("--keep-components", type=int, default=1,
                   help="passed to 2DGS's post_process_mesh as cluster_to_keep; 0 skips it")
    p.add_argument("--repo", default=os.environ.get("TWODGS", "2dgs"), help="2DGS checkout, for its mesh helper")
    args = p.parse_args()

    print(f"loading {args.mesh} ...", flush=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    lo = np.array([args.x_range[0], args.y_range[0], args.z_range[0]])
    hi = np.array([args.x_range[1], args.y_range[1], args.z_range[1]])
    print(f"  bounds min {np.round(mesh.get_min_bound(), 2)} max {np.round(mesh.get_max_bound(), 2)}")

    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=lo, max_bound=hi))
    print(f"  after crop: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces", flush=True)

    if args.keep_components > 0:
        mesh = load_post_process_mesh(args.repo)(mesh, cluster_to_keep=args.keep_components)
        print(f"  after 2DGS post_process_mesh: {len(mesh.vertices)} vertices, "
              f"{len(mesh.triangles)} faces")

    if len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_faces)
        print(f"  after decimation: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # collision only, so drop colour and keep the file small
    mesh.vertex_colors = o3d.utility.Vector3dVector()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.out, mesh, write_vertex_normals=False)
    print(f"wrote {args.out}")
    print(f"  final bounds min {np.round(mesh.get_min_bound(), 3)} max {np.round(mesh.get_max_bound(), 3)}")


if __name__ == "__main__":
    main()
