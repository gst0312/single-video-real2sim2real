"""Sample antipodal grasps on an object mesh, the way rsrd does it.

The sampler is jaxmp's `AntipodalGrasps.from_sample_mesh` (the official organ under
rsrd's `GraspablePart.from_mesh`); the wrapper logic here - 1000 surface samples, keep
the first `--max-grasps`, widen `--max-width` by 1.2x until at least one grasp exists -
is rsrd `rsrd/robot/graspable_obj.py:GraspablePart.from_mesh`, copied line for line.
rsrd is not imported because its module pulls the whole DiG/nerfstudio stack; the
wrapper is ten lines.

`--max-width` defaults to 0.08 m: our gripper is the Robotiq 2F-85 (85 mm stroke),
not the Franka hand rsrd's 0.04/0.045 was written for, and the mustard bottle's body
is 55-65 mm across - the demo grasp - which a 45 mm cap would exclude.

Outputs an npz with grasp centres and axes in the mesh frame, plus a glb of the mesh
with grasp markers for eyeballing.

    $TRAJ_VENV/bin/python scripts/datagen/sample_grasps.py \
        --mesh $GSWORLD_ROOT/assets/object_assets/mustard/mustard.obj \
        --out $WORK/traj/mustard_grasps.npz
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import trimesh

from jaxmp.extras.grasp_antipodal import AntipodalGrasps


def sample(mesh: trimesh.Trimesh, max_width: float, max_grasps: int):
    """rsrd GraspablePart.from_mesh, verbatim except for returning the width used."""
    grasps = AntipodalGrasps.from_sample_mesh(mesh, max_samples=1000, max_width=max_width)
    grasps = jax.tree.map(lambda x: x[:max_grasps], grasps)
    while len(grasps.centers) == 0:
        max_width *= 1.2
        grasps = AntipodalGrasps.from_sample_mesh(mesh, max_samples=1000, max_width=max_width)
        grasps = jax.tree.map(lambda x: x[:max_grasps], grasps)
    return grasps, max_width


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--max-width", type=float, default=0.08,
                   help="widest graspable span; Robotiq 2F-85 opens to 85 mm")
    p.add_argument("--max-grasps", type=int, default=20, help="rsrd keeps 20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    np.random.seed(args.seed)  # from_sample_mesh samples with numpy
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    print(f"{args.mesh}: {len(mesh.vertices)} vertices, extents {np.round(mesh.extents, 4)} m")

    grasps, used_width = sample(mesh, args.max_width, args.max_grasps)
    centers = np.asarray(grasps.centers)
    axes = np.asarray(grasps.axes)
    print(f"{len(centers)} grasps at max width {used_width:.3f} m")
    print(f"  centre span: {np.round(centers.min(0), 3)} .. {np.round(centers.max(0), 3)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, centers=centers, axes=axes, max_width=used_width, mesh=str(args.mesh))
    print(f"wrote {out}")

    # eyeball copy: the mesh plus a marker line per grasp along the finger axis
    scene = trimesh.Scene(mesh)
    for c, a in zip(centers, axes):
        seg = trimesh.load_path(np.stack([c - 0.03 * a, c + 0.03 * a]))
        scene.add_geometry(seg)
    glb = out.with_suffix(".glb")
    scene.export(glb)
    print(f"wrote {glb}")


if __name__ == "__main__":
    main()
