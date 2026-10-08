"""Score a rendered deployment view against the real frame, region by region.

The numbers are the ones 2DGS reports, computed with 2DGS's own code: PSNR from
`utils/image_utils.py` and SSIM from `utils/loss_utils.py`, so a value here means the same
thing as a value in their tables. Whole-image numbers hide what we care about, though, so
they are also computed over named boxes: the tabletop and the curtain, where floaters show
up, and each arm, where alignment shows up.

Two extra numbers per region, because they say directly what a splat gets wrong:

  brightness ratio  real over rendered, per channel, on the pixels both call bright. The
                    neighbour arm is the control: it is scene splat, so if the exposure
                    match is right its ratio is one.
  silhouette IoU    over bright pixels, with the best whole-pixel shift searched as well.
                    A high IoU that needs a shift means the geometry is right and the pose
                    is not; a low IoU at zero shift means the shape itself is wrong.

Boxes are given in the 1280x720 frame of the deployment camera and can be overridden.
"""

import argparse
import os
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REGIONS = {
    # y0, y1, x0, x1
    "table": (500, 650, 600, 1000),
    "curtain_left": (120, 480, 30, 380),
    "curtain_back": (30, 300, 830, 1030),
    "our_arm": (0, 420, 560, 860),
    "neighbour_arm": (40, 380, 1040, 1180),
    "whole": (0, 720, 0, 1280),
}


def load_2dgs_metrics(repo):
    """Import 2DGS's psnr and ssim without running the package __init__."""
    out = {}
    for module, names in (("utils/image_utils.py", ["psnr"]), ("utils/loss_utils.py", ["ssim"])):
        spec = importlib.util.spec_from_file_location(Path(module).stem, Path(repo) / module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in names:
            out[n] = getattr(mod, n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--twodgs", default=os.environ.get("TWODGS", "2dgs"))
    p.add_argument("--bright", type=float, default=110.0, help="a bright pixel, 0 to 255")
    p.add_argument("--shift", type=int, default=15, help="how far to search for a better shift")
    p.add_argument("--out", default=None)
    p.add_argument("--diff-out", default=None,
                   help="write real, rendered and the difference side by side, per region")
    p.add_argument("--diff-regions", nargs="*", default=["table", "curtain_left", "our_arm"])
    p.add_argument("--diff-gain", type=float, default=4.0)
    p.add_argument("--fit-exposure", nargs="*", default=None,
                   help="fit the per-channel gain and offset that carries the render onto the "
                        "real frame, over these regions (default: the table and both curtains, "
                        "which are scene splat and unaffected by how the robot is drawn). The "
                        "numbers go straight into relight_splat.py.")
    args = p.parse_args()

    metrics = load_2dgs_metrics(args.twodgs)
    real = np.asarray(Image.open(args.real).convert("RGB")).astype(np.float64)
    sim = np.asarray(Image.open(args.sim).convert("RGB")).astype(np.float64)
    if real.shape != sim.shape:
        raise SystemExit(f"{real.shape} against {sim.shape}")

    def iou(a, b):
        return float((a & b).sum() / max(1, (a | b).sum()))

    report = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        r, s = real[y0:y1, x0:x1], sim[y0:y1, x0:x1]
        # contiguous: 2DGS's psnr calls .view, which a transposed crop cannot do
        tr = torch.tensor(np.ascontiguousarray(r.transpose(2, 0, 1) / 255.0))[None]
        ts = torch.tensor(np.ascontiguousarray(s.transpose(2, 0, 1) / 255.0))[None]
        entry = {
            "psnr": float(metrics["psnr"](ts, tr).mean()),
            "ssim": float(metrics["ssim"](ts, tr)),
            "mean_abs_error": float(np.abs(r - s).mean()),
        }
        mr, ms = r.mean(2) > args.bright, s.mean(2) > args.bright
        both = mr & ms
        if both.sum() > 200:
            entry["bright_pixels"] = [int(mr.sum()), int(ms.sum())]
            entry["brightness_ratio"] = [float(v) for v in r[both].mean(0) / s[both].mean(0)]
            entry["iou"] = iou(mr, ms)
            best = max((iou(mr, np.roll(np.roll(ms, dy, 0), dx, 1)), dx, dy)
                       for dy in range(-args.shift, args.shift + 1)
                       for dx in range(-args.shift, args.shift + 1))
            entry["iou_best"] = best[0]
            entry["iou_best_shift"] = [best[1], best[2]]
        if args.diff_out and name in args.diff_regions:
            gap = np.full((y1 - y0, 6, 3), 255.0)
            diff = np.clip(np.abs(r - s) * args.diff_gain, 0, 255)
            row = np.concatenate([r, gap, s, gap, diff], axis=1)
            out = Path(args.diff_out)
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(row.astype("uint8")).save(out / f"{name}.png")

        report[name] = entry
        line = f"{name:14s} psnr {entry['psnr']:5.2f}  ssim {entry['ssim']:.4f}  mae {entry['mean_abs_error']:5.2f}"
        if "iou" in entry:
            line += (f"  iou {entry['iou']:.3f} (best {entry['iou_best']:.3f} at "
                     f"{entry['iou_best_shift']})  bright ratio "
                     + " ".join(f"{v:.3f}" for v in entry["brightness_ratio"]))
        print(line)

    if args.fit_exposure is not None:
        names = args.fit_exposure or ["table", "curtain_left", "curtain_back"]
        r = np.concatenate([real[y0:y1, x0:x1].reshape(-1, 3)
                            for y0, y1, x0, x1 in (REGIONS[n] for n in names)])
        s = np.concatenate([sim[y0:y1, x0:x1].reshape(-1, 3)
                            for y0, y1, x0, x1 in (REGIONS[n] for n in names)])
        gain, offset = [], []
        for c in range(3):
            a = np.stack([s[:, c], np.ones(len(s))], axis=1)
            g, o = np.linalg.lstsq(a, r[:, c], rcond=None)[0]
            gain.append(float(g))
            offset.append(float(o))
        report["fitted_exposure"] = {"regions": names, "gain": gain, "offset": offset}
        print("fit over " + ", ".join(names) + ":\n  --gain "
              + " ".join(f"{v:.4f}" for v in gain)
              + " --offset " + " ".join(f"{v:.2f}" for v in offset))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
