"""Convert GSWorld's 3DGS scene splat into a 2DGS asset PolaRiS can load, in the robot
base frame.

Their `fr3_final.ply` is a true 3DGS: the smallest axis falls on scale_0/1/2 roughly a
third of the time each, and PolaRiS's loader keeps `sorted(scale_names)[:2]`, so feeding
it three columns unsorted would corrupt most shapes. Their frame differs from the robot
base frame by the rigid `sim2gs_arm_trans` in their config. Three things happen here:

1. Rigid transform into the base frame, the inverse of `sim2gs_arm_trans`. Positions and
   per-gaussian rotations move the usual way. The higher-order spherical harmonics live
   on world axes and must rotate with the scene; that rotation is `transform_shs`,
   copied verbatim from SplatSim (github.com/qureshinomaan/SplatSim, ea07395,
   `splatsim/utils/robot_splat_render_utils.py`) - e3nn Wigner-D blocks behind a
   yzx->xyz basis permutation. SplatSim runs the same function every frame to pose
   robot link splats. The DC term is direction independent and stays put.

2. 3DGS -> 2DGS. Per gaussian: drop the smallest of the three axes, permute the two
   kept axes into columns 0 and 1. When (i, j, k) is an odd permutation the determinant
   flips, so the third column is negated to stay a rotation; a disc's normal sign does
   not render. Two scale columns are written, like the official Robotiq link plys, so
   the loader's "first two" is unambiguous.

3. Optionally, put the ArUco markers back. `fr3_final` hides every gaussian within the
   two marker discs (opacity forced to -15) and lays 185 faint cloth patches over them;
   colours were never touched, so restoring the opacities from `fr3_scene06_clean` and
   hiding the patches brings the markers back as alignment cues.

Self checks, all on by default: config rotation orthonormality; SH values at matched
view directions before and after the rotation (2DGS's own eval_sh); kept scales equal
the two largest original axes; disc normals parallel to the original smallest axis.

Runs in the 2dgs venv (needs torch, e3nn, einops, plyfile, scipy):
    $TWODGS/.venv/bin/python scripts/real2sim/gsworld_splat_to_2dgs.py \
        --splat .../fr3_final.ply --config .../fr3_robotiq_final.json \
        --semantics .../fr3_final_semantics_gs.npy \
        --restore-markers-from .../fr3_scene06_clean.ply \
        --markers-json $WORK/markers_in_base.json \
        --out-prefix $WORK/gs06
"""

import argparse
import os
import importlib.util
import json
from pathlib import Path

import einops
import numpy as np
import torch
from e3nn import o3
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def transform_shs(shs_feat, rotation_matrix):
    """SplatSim's SH rotation, copied verbatim (robot_splat_render_utils.py, ea07395)."""

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], device=rotation_matrix.device).float() # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    rot_angles = (rot_angles[0].cpu(), rot_angles[1].cpu(), rot_angles[2].cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    device = shs_feat.device
    dtype = shs_feat.dtype

    D = torch.block_diag(D_1.to(device=device, dtype=dtype),
                         D_2.to(device=device, dtype=dtype),
                         D_3.to(device=device, dtype=dtype))                    # (15,15)

    # take l=1..3 bands and move RGB before SH for a single einsum
    sh = einops.rearrange(shs_feat[..., :15, :], '... s r -> ... r s')  # (..., 3, 15)
    sh_rot = torch.einsum('...ij, ...rj -> ...ri', D, sh)               # (..., 3, 15)
    shs_feat[..., :15, :] = einops.rearrange(sh_rot, '... r i -> ... i r')  # (..., 15, 3)

    return shs_feat


def load_eval_sh(repo):
    """2DGS's own eval_sh, imported by file path without running the package."""
    path = Path(repo) / "utils" / "sh_utils.py"
    spec = importlib.util.spec_from_file_location("sh_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.eval_sh


def quat_wxyz_to_mat(q):
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def mat_to_quat_wxyz(m):
    return Rotation.from_matrix(m).as_quat()[:, [3, 0, 1, 2]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True, help="GSWorld 3DGS ply, in their GS frame")
    p.add_argument("--config", required=True, help="their scene json with sim2gs_arm_trans")
    p.add_argument("--semantics", default=None, help="per-gaussian labels, copied through")
    p.add_argument("--restore-markers-from", default=None,
                   help="the clean ply whose opacities bring the hidden ArUcos back")
    p.add_argument("--markers-json", default=None,
                   help="markers_in_base.json with the two marker centres in the base frame")
    p.add_argument("--marker-radius", type=float, default=0.15,
                   help="restore opacities within this planar radius of a marker centre; "
                        "their paint-out killed the marker and the cloth around it out to "
                        "the quad edge, so restoring only the disc would leave a bare ring")
    p.add_argument("--marker-disc-radius", type=float, default=0.075,
                   help="the printed marker itself; the taller restore window applies here")
    p.add_argument("--marker-slab-below", type=float, default=0.02,
                   help="restore this far below the nominal marker plane")
    p.add_argument("--marker-slab-above", type=float, default=0.07,
                   help="above the plane, inside the disc only. Asymmetric because the "
                        "reconstructed marker surfaces sit 20-60 mm above our nominal plane "
                        "in their frame; everything above 70 mm is genuine floater glow and "
                        "stays dead")
    p.add_argument("--marker-ring-slab", type=float, default=0.02,
                   help="height window for the cloth ring outside the disc; anything they "
                        "killed higher up there is hovering glow, and restoring it smears "
                        "into blotches at the deployment camera's grazing angle")
    p.add_argument("--extra-transform", default=None,
                   help="optional npz with 'rot' (3,3) and 'trans' (3,), applied after the "
                        "config transform; used for the rigid base-frame refinement")
    p.add_argument("--keep-3d", action="store_true",
                   help="skip the 2DGS squash and write all three scales; for the "
                        "gsplat-3dgs-renderer backend, which takes true 3DGS")
    p.add_argument("--twodgs", default=os.environ.get("TWODGS", "2dgs"))
    p.add_argument("--sh-check-samples", type=int, default=2048)
    p.add_argument("--out-prefix", required=True,
                   help="writes <prefix>_2dgs.ply, <prefix>_semantics.npy, <prefix>_report.json")
    args = p.parse_args()

    report = {}
    ply = PlyData.read(args.splat)
    vert = ply["vertex"].data.copy()
    n = len(vert)
    report["gaussians"] = n
    print(f"{n} gaussians from {args.splat}")

    cfg = json.load(open(args.config))
    T = np.array(cfg["sim2gs_arm_trans"], dtype=np.float64)
    R_cfg, t_cfg = T[:3, :3], T[:3, 3]
    orth = float(np.abs(R_cfg @ R_cfg.T - np.eye(3)).max())
    det = float(np.linalg.det(R_cfg))
    report["config_rotation"] = {"orthonormality_error": orth, "det": det}
    print(f"sim2gs rotation: |RR^T - I| max {orth:.2e}, det {det:.6f}")
    if orth > 1e-3 or abs(det - 1) > 1e-3:
        raise SystemExit("sim2gs_arm_trans is not a rigid transform; refusing to continue")

    # the rotation the scene undergoes: GS frame -> base frame, then any refinement
    R_apply = R_cfg.T
    t_apply = -R_cfg.T @ t_cfg
    if args.extra_transform:
        d = np.load(args.extra_transform)
        R_e, t_e = np.asarray(d["rot"], np.float64), np.asarray(d["trans"], np.float64)
        R_apply = R_e @ R_apply
        t_apply = R_e @ t_apply + t_e
        report["extra_transform"] = {"rot": R_e.tolist(), "trans": t_e.tolist()}
        print("composed extra rigid refinement into the transform")

    # markers back first, while positions still match the clean file byte for byte
    if args.restore_markers_from:
        clean = PlyData.read(args.restore_markers_from)["vertex"].data
        def poskey(v):
            return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32).view("V12").ravel()
        markers = json.load(open(args.markers_json))
        centres_base = np.array([m["pos"] for m in markers["markers"]], np.float64)
        # flat cylinders in the base frame, because their kill was a tall slab over each
        # marker and the floaters it removed above the table should stay removed
        xyz_gs = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
        xyz_base_probe = (xyz_gs - t_cfg) @ R_cfg
        near = np.zeros(n, bool)
        for c in centres_base:
            planar = np.linalg.norm(xyz_base_probe[:, :2] - c[:2], axis=1)
            dz = xyz_base_probe[:, 2] - c[2]
            disc = (planar < args.marker_disc_radius) & \
                   (dz > -args.marker_slab_below) & (dz < args.marker_slab_above)
            ring = (planar >= args.marker_disc_radius) & (planar < args.marker_radius) & \
                   (np.abs(dz) < args.marker_ring_slab)
            near |= disc | ring
        common, i_clean, i_here = np.intersect1d(poskey(clean), poskey(vert),
                                                 return_indices=True)
        from_clean = np.zeros(n, dtype=np.int64) - 1
        from_clean[i_here] = i_clean
        restore = near & (from_clean >= 0) & (vert["opacity"] != clean["opacity"][np.clip(from_clean, 0, None)])
        vert["opacity"][restore] = clean["opacity"][from_clean[restore]]
        # the cover patches are the gaussians that exist only in this file; hide every one
        # near a marker, wherever it sits in height
        patch_planar = np.full(n, np.inf)
        for c in centres_base:
            patch_planar = np.minimum(
                patch_planar, np.linalg.norm(xyz_base_probe[:, :2] - c[:2], axis=1))
        patch = (from_clean < 0) & (patch_planar < 0.25)
        vert["opacity"][patch] = -15.0
        report["markers"] = {"restored_opacities": int(restore.sum()),
                             "patches_hidden": int(patch.sum())}
        print(f"markers: restored {int(restore.sum())} opacities, hid {int(patch.sum())} patch gaussians")

    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    quats = np.stack([vert[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    s_log = np.stack([vert[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
    rest = np.stack([vert[f"f_rest_{i}"] for i in range(45)], 1).astype(np.float64)
    dc = np.stack([vert[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)

    # positions and per-gaussian rotations
    xyz_new = xyz @ R_apply.T + t_apply
    q_apply = mat_to_quat_wxyz(R_apply[None])[0]
    w1, x1, y1, z1 = q_apply
    w2, x2, y2, z2 = quats.T
    quats_new = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)

    # spherical harmonics: (N, 45) channel-major -> (N, 15, 3), rotate, back
    sh_before = rest.reshape(n, 3, 15).transpose(0, 2, 1).copy()
    sh_t = torch.from_numpy(sh_before.copy()).float()
    transform_shs(sh_t, torch.from_numpy(R_apply).float())
    sh_after = sh_t.numpy().astype(np.float64)
    rest_new = sh_after.transpose(0, 2, 1).reshape(n, 45)

    # self check: the rotated SH evaluated along the rotated direction must give the
    # colour the original SH gave along the original direction
    eval_sh = load_eval_sh(args.twodgs)
    rng = np.random.default_rng(0)
    pick = rng.choice(n, min(args.sh_check_samples, n), replace=False)
    dirs = rng.normal(size=(len(pick), 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sh_full_before = np.concatenate([dc[pick, :, None],
                                     sh_before[pick].transpose(0, 2, 1)], axis=2)
    sh_full_after = np.concatenate([dc[pick, :, None],
                                    sh_after[pick].transpose(0, 2, 1)], axis=2)
    col_before = eval_sh(3, torch.from_numpy(sh_full_before), torch.from_numpy(dirs))
    col_after = eval_sh(3, torch.from_numpy(sh_full_after),
                        torch.from_numpy(dirs @ R_apply.T))
    sh_err = float((col_after - col_before).abs().max())
    spread = float((col_before - col_before.mean(0)).abs().max())
    report["sh_rotation_check"] = {"max_abs_error": sh_err, "colour_spread": spread}
    print(f"SH rotation check: max |after(Rd) - before(d)| = {sh_err:.2e} "
          f"(colour spread {spread:.2f})")
    if sh_err > 1e-4 * max(1.0, spread):
        raise SystemExit("SH rotation does not commute with the scene rotation")

    if args.keep_3d:
        n_scales = 3
        s_kept = s_log
        quats_out = quats_new
    else:
        n_scales = 2
        # 3DGS -> 2DGS: sort out the smallest axis
        k = s_log.argmin(1)
        others = np.sort(np.stack([(k + 1) % 3, (k + 2) % 3], 1), 1)
        perm = np.concatenate([others, k[:, None]], 1)      # (N, 3): kept i < j, then k
        rows = np.arange(n)[:, None]
        s_kept = s_log[rows, perm[:, :2]]

        R_g = quat_wxyz_to_mat(quats_new)
        R_new = np.take_along_axis(R_g, perm[:, None, :], axis=2).copy()
        odd = k == 1                                        # (0,2,1) is the odd one
        R_new[odd, :, 2] *= -1
        quats_out = mat_to_quat_wxyz(R_new)

        dets = np.linalg.det(R_new)
        report["squash"] = {
            "min_axis_on_scale_2_fraction": float((k == 2).mean()),
            "det_error_max": float(np.abs(dets - 1).max()),
        }
        print(f"squash: min axis was already scale_2 for {(k == 2).mean():.1%}, "
              f"max |det - 1| = {np.abs(dets - 1).max():.2e}")

        # self check: kept scales are the two largest, disc normal the old smallest axis
        top2 = np.sort(s_log, 1)[:, 1:]
        assert np.allclose(np.sort(s_kept, 1), top2), "kept scales are not the two largest"
        normal_dot = np.abs(np.einsum("ni,ni->n", R_new[:, :, 2], R_g[rows[:, 0], :, k]))
        report["squash"]["normal_alignment_min"] = float(normal_dot.min())
        assert normal_dot.min() > 1 - 1e-6, "disc normal does not match the dropped axis"

    out = np.zeros(n, dtype=[(name, "f4") for name in
                             ["x", "y", "z", "nx", "ny", "nz"]
                             + [f"f_dc_{i}" for i in range(3)]
                             + [f"f_rest_{i}" for i in range(45)]
                             + ["opacity"]
                             + [f"scale_{i}" for i in range(n_scales)]
                             + [f"rot_{i}" for i in range(4)]])
    out["x"], out["y"], out["z"] = xyz_new.T
    for i in range(3):
        out[f"f_dc_{i}"] = dc[:, i]
    for i in range(45):
        out[f"f_rest_{i}"] = rest_new[:, i]
    out["opacity"] = vert["opacity"]
    for i in range(n_scales):
        out[f"scale_{i}"] = s_kept[:, i]
    for i in range(4):
        out[f"rot_{i}"] = quats_out[:, i]

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_3dgs.ply" if args.keep_3d else "_2dgs.ply"
    PlyData([PlyElement.describe(out, "vertex")]).write(str(prefix) + suffix)
    print(f"wrote {prefix}{suffix}")
    if args.semantics:
        sem = np.load(args.semantics)
        assert len(sem) == n
        np.save(str(prefix) + "_semantics.npy", sem)
    ext = xyz_new.min(0), xyz_new.max(0)
    report["base_frame_extent"] = {"min": [float(v) for v in ext[0]],
                                   "max": [float(v) for v in ext[1]]}
    print(f"base-frame extent: {np.round(ext[0], 3)} .. {np.round(ext[1], 3)}")
    with open(str(prefix) + "_report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
