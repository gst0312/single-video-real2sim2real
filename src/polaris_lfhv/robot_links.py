"""Helpers shared by the scripts that split a splat into per-link robot assets.

The split follows SplatSim (github.com/qureshinomaan/SplatSim,
`scripts/articulated_robot_pipeline.py`): sample points on one link at a time, label them,
fit a k-nearest-neighbour classifier, and let it say which link each gaussian belongs to.
SplatSim samples the visible link by rendering depth in PyBullet; we read the meshes out of
the USD, which is exact and keeps every frame consistent with the articulation PolaRiS
evaluates with.

One trap, verified on this machine: with `use_fabric=True` IsaacLab writes link poses to
Fabric and never back to the USD stage, so `UsdGeom.XformCache` returns the default pose no
matter what joint state has been written. Posing the robot and reading mesh transforms from
USD therefore silently gives the same answer every time. Link poses must come from
`robot.data.body_pos_w` / `body_quat_w`; what USD is good for is the transform from a body
to the meshes under it, which is static and pose independent.
"""

import numpy as np


def quat_to_mat(q):
    """Rotation matrix from a w,x,y,z quaternion, matching IsaacLab's convention."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def sample_mesh(points, counts, indices, n, rng):
    """Uniform points over a USD mesh, weighted by triangle area.

    USD stores faces as vertex counts plus a flat index list, so polygons are fanned into
    triangles first.
    """
    tris = []
    k = 0
    for c in counts:
        fan = indices[k:k + c]
        for i in range(1, c - 1):
            tris.append((fan[0], fan[i], fan[i + 1]))
        k += c
    tris = np.array(tris)
    if not len(tris):
        return np.zeros((0, 3))
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3))
    pick = rng.choice(len(tris), size=n, p=area / area.sum())
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    over = (u + v) > 1
    u[over] = 1 - u[over]
    v[over] = 1 - v[over]
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])


def collect_meshes(stage, robot, root="/World/envs/env_0/robot", samples=2000, seed=0):
    """Sample every robot mesh, in the frame of the body that carries it.

    Returns a list of dicts with the prim path, the asset name PolaRiS expects (the prim
    path with slashes turned into dashes), the index of the owning body in
    `robot.data.body_names`, and the sampled points expressed in that body's frame.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rng = np.random.default_rng(seed)
    bodies = list(robot.data.body_names)
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root) or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        rel = path[len(root) + 1:]
        # The owning body is the nearest ancestor that is a body. Ancestors only: the
        # Panda meshes are named after their link (panda_link0/geometry/panda_link0), so
        # searching the whole path finds the mesh itself, the relative transform collapses
        # to identity, and the 0.01 scale sitting on the geometry prim is lost — the
        # samples then come out in centimetres read as metres.
        parts = rel.split("/")
        ancestors = parts[:-1]
        body = next((b for b in ancestors[::-1] if b in bodies), None)
        if body is None:
            continue
        depth = len(ancestors) - ancestors[::-1].index(body)
        body_prim = stage.GetPrimAtPath(f"{root}/" + "/".join(parts[:depth]))
        m = np.array(cache.ComputeRelativeTransform(prim, body_prim)[0], dtype=float).T
        local = np.array(pts, dtype=float)
        in_body = local @ m[:3, :3].T + m[:3, 3]
        span = in_body.max(axis=0) - in_body.min(axis=0)
        if span.max() > 1.0:
            raise SystemExit(
                f"{rel} spans {np.round(span, 2)} m in the frame of {body}; a link is never "
                "that large, so the mesh-to-body transform is wrong (a lost scale factor "
                "gives exactly this)")
        s = sample_mesh(in_body, np.array(counts), np.array(idx), samples, rng)
        if not len(s):
            continue
        out.append({"prim": rel, "name": rel.replace("/", "-"), "body": bodies.index(body),
                    "body_name": body, "points": s,
                    "mesh_in_body": m})
    return out


def body_frames(robot, index=0):
    """World poses of every body, as (positions, rotation matrices)."""
    pos = robot.data.body_pos_w[index].detach().cpu().numpy()
    quat = robot.data.body_quat_w[index].detach().cpu().numpy()
    rot = np.stack([quat_to_mat(q) for q in quat])
    return pos, rot
