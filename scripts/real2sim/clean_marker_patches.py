"""Finish the ArUco paint-out on the marker-free scene: kill what still reads as marker,
clone clean cloth over the area.

GSWorld's own `scene06_no_aruco_paintout.json` replace_regions does exactly this on their
side - their final differs from fr3_scene06_clean ONLY here: per marker an axis-aligned
QUAD (half-width 0.14 / 0.145, kill slab -0.15..0.12) is opacity-hidden and 185 sparse
donor clones are appended. Those patches read as a dark smudge in our deployment render
once the relight gain (1.46-1.6) amplifies the contrast, so this script replaces the same
two regions again, wider, on our converted asset where the marker centres are known in
metres to a few millimetres:

  1. inside the kill square, hide (opacity -15) every gaussian - with `--replace-all`
     that includes their donor clones and everything else visible there;
  2. clone the gaussians of a same-sized donor square picked from clean cloth, shift them
     onto the marker centre, and append them; the per-gaussian semantics file is extended
     with background labels so row alignment survives.

All footprints are squares (Chebyshev distance), not discs: their quads are squares, and
the first version's 0.13 disc left their corners (reaching 0.145*sqrt(2) = 0.205 from the
centre) alive, sampled its cloth statistics ring at 0.15-0.21 right on top of those
corners, and cloned from a donor disc that overlapped their patch - which is exactly the
smudge archipelago the deployment render showed. The donor must also be the same size as
the kill, or the killed rim is left as a hole for the dark background to show through.

Runs on the scene splat after the robot split (background only left).
"""

import argparse
import os
import json

import numpy as np
from plyfile import PlyData, PlyElement

p = argparse.ArgumentParser()
p.add_argument("--splat", required=True, help="scene splat, robot already split away")
p.add_argument("--semantics", required=True)
p.add_argument("--markers-json", default=os.path.join(os.environ.get("WORK", "work"), "markers_in_base.json"))
p.add_argument("--kill-half", type=float, default=0.165,
               help="half-width of the kill square; must exceed GSWorld's own quads "
                    "(0.145 half) so their donor patches and hidden rims go too")
p.add_argument("--slab", type=float, nargs=2, default=[-0.10, 0.13],
               help="height window around the marker centre; GSWorld's kill slab "
                    "reaches 0.12 above the marker plane")
p.add_argument("--annulus", type=float, nargs=2, default=[0.18, 0.26],
               help="local cloth statistics come from this square ring, clear of the "
                    "kill square and of their patch corners")
p.add_argument("--sigma", type=float, default=2.0,
               help="hide gaussians whose colour is this many sigmas off the cloth")
p.add_argument("--replace-all", action="store_true",
               help="hide everything visible in the kill square instead of only colour "
                    "outliers, GSWorld's own quad treatment; the statistical kill leaves "
                    "structural residue (odd opacity and scale) that still reads as a smudge")
p.add_argument("--donor-offset", type=float, nargs="*", default=[0.25, 0.10, 0.30, -0.05],
               help="x y pairs per marker: where to take the replacement cloth from. Both "
                    "point into the object placement area (clean, well-observed cloth), "
                    "clear of both marker quads and of the table cable")
p.add_argument("--donor-half", type=float, default=0.17,
               help="half-width of the donor square; at least the kill half, or the "
                    "killed rim is left as a hole")
p.add_argument("--recolor", action="store_true",
               help="instead of kill-and-clone: keep the marker's own gaussians where they "
                    "are and paint them cloth. The paper lies flat on the cloth plane, so "
                    "its gaussians are exactly where cloth belongs; recolouring in place "
                    "sidesteps the moved-cloth problem entirely - cloth carried in from "
                    "elsewhere renders with the view-dependent appearance of its old "
                    "position, as a hard-edged tinted pane when moved as a tile (both our "
                    "clones and GSWorld's own patches fail this way) and as colour specks "
                    "when scattered gaussian by gaussian (also tried). Use on the "
                    "marker-restored asset, not on their painted-out final")
p.add_argument("--recolor-sigma", type=float, default=1.5,
               help="recolour gaussians this many sigmas off the ring cloth")
p.add_argument("--recolor-slab", type=float, nargs=2, default=[-0.06, 0.10],
               help="height window for recolouring: the paper and the glow above it "
                    "(GSWorld's own kill slab reaches 0.12)")
p.add_argument("--recolor-knn", type=int, default=25,
               help="each recoloured gaussian takes its colour statistics from this many "
                    "nearest ordinary cloth gaussians")
p.add_argument("--recolor-jitter", type=float, default=0.15,
               help="luminance spread of the recoloured rows, in local-sigma units; the "
                    "0.5 first tried reads as fold-like streaks from the wrist camera "
                    "where the real cloth is smooth")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--semantics-out", required=True)
args = p.parse_args()

C0 = 0.28209479177387814

ply = PlyData.read(args.splat)
data = ply["vertex"].data.copy()
sem = np.load(args.semantics)
assert len(sem) == len(data), f"{len(sem)} labels for {len(data)} gaussians"
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
colour = dc * C0 + 0.5
opa = data["opacity"].astype(np.float64)

markers = json.load(open(args.markers_json))["markers"]
donors = np.array(args.donor_offset, np.float64).reshape(-1, 2)
rng = np.random.default_rng(args.seed)
new_rows = []
new_sem = []
for m, donor_off in zip(markers, donors):
    c = np.array(m["pos"], np.float64)
    planar = np.abs(xyz[:, :2] - c[:2]).max(1)   # Chebyshev: square footprints
    dz = xyz[:, 2] - c[2]
    in_slab = (dz > args.slab[0]) & (dz < args.slab[1])
    disc = (planar < args.kill_half) & in_slab
    ring = (planar >= args.annulus[0]) & (planar < args.annulus[1]) & in_slab & (opa > -3)
    mu = colour[ring].mean(0)
    sd = colour[ring].std(0) + 1e-6
    if args.recolor:
        from scipy.spatial import cKDTree

        # sub-visible rows are hidden outright: rows at opacity <= -6 (0.25% alpha) are
        # individually invisible but a grazing wrist ray crosses hundreds and their
        # untouched SH stacks into haze. -15 (3e-7) makes them truly gone.
        faint = (planar < args.kill_half) & (dz > args.recolor_slab[0]) & \
                (dz < args.recolor_slab[1]) & (opa <= -6)
        data["opacity"][faint] = -15.0
        # recolour everything visible in the square, not only colour outliers: rows whose
        # DC reads as cloth can still carry wild SH (paper glare) that the DC test cannot
        # see
        tight = (planar < args.kill_half) & (dz > args.recolor_slab[0]) & \
                (dz < args.recolor_slab[1]) & (opa > -6)
        off = np.abs((colour - mu) / sd).max(1)
        sel = tight
        n_sel = int(sel.sum())
        # colour from the nearest ordinary cloth, not one ring-wide mean: the cloth is
        # unevenly lit and a constant fill reads as a faint square under the gradient.
        # Surface band only - the slab above holds glow gaussians whose SH turns the
        # recoloured area into a rainbow smear from the wrist camera's angles.
        ref = (ring | (tight & ~sel)) & (off <= args.recolor_sigma) & (np.abs(dz) < 0.02)
        tree = cKDTree(xyz[ref, :2])
        _, nn = tree.query(xyz[sel, :2], k=args.recolor_knn)
        ref_colour = colour[ref]
        local_mu = ref_colour[nn].mean(1)
        local_sd = ref_colour[nn].std(1)
        # one luminance draw per row, never per channel: independent RGB jitter gives
        # every gaussian its own slight hue, which cancels in the far view but shows as
        # pastel rainbow streaks when the wrist camera grazes single large gaussians
        fresh = np.clip(local_mu + rng.normal(0.0, 1.0, (n_sel, 1)) * local_sd
                        * args.recolor_jitter, 0.02, 0.98)
        for i in range(3):
            data[f"f_dc_{i}"][sel] = (fresh[:, i] - 0.5) / C0
        # diffuse only: synthesised per-row mean SH holds near the training views but
        # explodes into rainbow bands from the wrist camera's grazing directions; the
        # cloth is matte, so dropping the view dependence is the honest approximation
        for k in range(45):
            data[f"f_rest_{k}"][sel] = 0.0
        print(f"marker at {np.round(c[:2], 3)}: recoloured {n_sel} from {int(ref.sum())} "
              f"cloth refs, hid {int(faint.sum())} sub-visible rows")
        continue
    if args.replace_all:
        kill = disc & (opa > -6)
    else:
        off = np.abs((colour - mu) / sd).max(1)
        kill = disc & (opa > -6) & (off > args.sigma)
    data["opacity"][kill] = -15.0

    donor_c = c[:2] + donor_off
    donor = (np.abs(xyz[:, :2] - donor_c).max(1) < args.donor_half) & \
            in_slab & (opa > -3)
    clone = data[donor].copy()
    clone["x"] += np.float32(c[0] - donor_c[0])
    clone["y"] += np.float32(c[1] - donor_c[1])
    # the cloth plane is tilted ~1.3 deg against the base, so cloth carried sideways must
    # also move down/up to stay ON the plane - without this a far donor hovers ~8 mm above
    # the local cloth and reads as a hard-edged pane at the grazing view. The cloth surface
    # is the low envelope of each region's gaussians, floaters sit above it.
    dz_plane = np.quantile(xyz[ring, 2], 0.2) - np.quantile(xyz[donor, 2], 0.2)
    clone["z"] += np.float32(dz_plane)
    # the cloth is not evenly lit; match the clones to the ring around the marker, or the
    # patch reads as a tinted disc
    donor_mean = colour[donor].mean(0)
    for i in range(3):
        gain = mu[i] / max(donor_mean[i], 1e-6)
        clone[f"f_dc_{i}"] = ((colour[donor][:, i] * gain) - 0.5) / C0
        for j in range(15):  # the channel's whole SH band scales with it
            clone[f"f_rest_{i * 15 + j}"] *= gain

    new_rows.append(clone)
    new_sem.append(np.full(len(clone), -1, dtype=sem.dtype))
    print(f"marker at {np.round(c[:2], 3)}: killed {int(kill.sum())} of {int(disc.sum())} "
          f"in the square (cloth ref {int(ring.sum())}), cloned {len(clone)} from "
          f"{np.round(donor_c, 3)}, plane dz {dz_plane * 1000:+.1f} mm")

out_data = np.concatenate([data] + new_rows)
out_sem = np.concatenate([sem] + new_sem)
PlyData([PlyElement.describe(out_data, "vertex")]).write(args.out)
np.save(args.semantics_out, out_sem)
print(f"wrote {args.out} ({len(out_data)} gaussians) and {args.semantics_out}")
