"""Turn a folder of real-vs-sim frames into one table.

Reads what `compare_to_real_rollout.py --report` wrote for each segment (which joint state
each frame was paired with, and how far apart the two clocks were) and scores every image
pair the same way `measure_render_error.py` does, in one process, so eight segments cost
one import of the metrics rather than dozens.

    python scripts/real2sim/summarise_rollout_pack.py --pack .../rollout_pack --out .../summary.json
"""

import argparse
import os
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--twodgs", default=os.environ.get("TWODGS", "2dgs"))
    p.add_argument("--bright", type=float, default=110.0,
                   help="a pixel this bright counts as robot, for the silhouette overlap")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    mre = load(HERE / "measure_render_error.py", "mre")
    metrics = mre.load_2dgs_metrics(args.twodgs)
    pack = Path(args.pack)

    def score(real, sim, box):
        y0, y1, x0, x1 = box
        a = torch.from_numpy(real[y0:y1, x0:x1].astype(np.float32) / 255).permute(2, 0, 1)
        b = torch.from_numpy(sim[y0:y1, x0:x1].astype(np.float32) / 255).permute(2, 0, 1)
        return {"psnr": float(metrics["psnr"](a, b).mean()),
                "ssim": float(metrics["ssim"](a[None], b[None])),
                "mae": float(np.abs(real[y0:y1, x0:x1].astype(np.float64)
                                    - sim[y0:y1, x0:x1].astype(np.float64)).mean())}

    def iou(real, sim, box):
        y0, y1, x0, x1 = box
        a = real[y0:y1, x0:x1].mean(2) > args.bright
        b = sim[y0:y1, x0:x1].mean(2) > args.bright
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else float("nan")

    out = {"segments": {}}
    for report in sorted(pack.glob("*.json")):
        if report.name == Path(args.out).name:
            continue
        rep = json.loads(report.read_text())
        seg = rep["segment"]
        rows = []
        for fr in rep["frames"]:
            row = {"frame": fr["frame"], "pair_gap_ms": fr["pair_gap_ms"],
                   "wrist_pair_gap_ms": fr["wrist_pair_gap_ms"],
                   "qpos_deg": [round(float(np.degrees(v)), 2) for v in fr["qpos"]],
                   "qpos_write_error_deg": float(np.degrees(np.abs(
                       np.array(fr["achieved_qpos"]) - np.array(fr["qpos"]))).max())}
            for cam in ("external_cam", "wrist_cam"):
                stem = f'{seg}_{fr["frame"]:06d}_{cam}'
                rp, sp = pack / f"{stem}_real.png", pack / f"{stem}_sim.png"
                if not rp.exists():
                    continue
                real = np.asarray(Image.open(rp).convert("RGB"))
                sim = np.asarray(Image.open(sp).convert("RGB"))
                entry = {"whole": score(real, sim, mre.REGIONS["whole"]),
                         "bright_ratio": float(sim.mean() / real.mean())}
                if cam == "external_cam":
                    for name in ("table", "curtain_left", "our_arm", "neighbour_arm"):
                        entry[name] = score(real, sim, mre.REGIONS[name])
                    entry["our_arm"]["iou"] = iou(real, sim, mre.REGIONS["our_arm"])
                    entry["neighbour_arm"]["iou"] = iou(real, sim, mre.REGIONS["neighbour_arm"])
                    entry["whole"]["iou"] = iou(real, sim, mre.REGIONS["whole"])
                row[cam] = entry
            rows.append(row)
        out["segments"][seg] = rows

    # printed table: one line per segment, averaged over its frames
    print(f"{'segment':14s} {'frames':>6s} {'pair gap ms':>11s} "
          f"{'ext PSNR':>9s} {'SSIM':>6s} {'IoU':>6s} {'bright':>6s} "
          f"{'wrist PSNR':>10s} {'SSIM':>6s} {'bright':>6s}")
    agg = {}
    for seg, rows in out["segments"].items():
        def mean(path):
            vals = []
            for r in rows:
                v = r
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                    if v is None:
                        break
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        agg[seg] = {
            "gap": mean(["pair_gap_ms"]),
            "e_psnr": mean(["external_cam", "whole", "psnr"]),
            "e_ssim": mean(["external_cam", "whole", "ssim"]),
            "e_iou": mean(["external_cam", "whole", "iou"]),
            "e_bright": mean(["external_cam", "bright_ratio"]),
            "w_psnr": mean(["wrist_cam", "whole", "psnr"]),
            "w_ssim": mean(["wrist_cam", "whole", "ssim"]),
            "w_bright": mean(["wrist_cam", "bright_ratio"]),
            "arm_iou": mean(["external_cam", "our_arm", "iou"]),
            "qpos_err": mean(["qpos_write_error_deg"]),
        }
        a = agg[seg]
        print(f'{seg:14s} {len(rows):3d} {a["gap"]:10.1f} '
              f'{a["e_psnr"]:9.2f} {a["e_ssim"]:6.3f} {a["e_iou"]:6.3f} {a["e_bright"]:6.3f} '
              f'{a["w_psnr"]:8.2f} {a["w_ssim"]:6.3f} {a["w_bright"]:6.3f}')
    print(f'{"mean over all":14s} {sum(len(r) for r in out["segments"].values()):3d} '
          f'{np.mean([a["gap"] for a in agg.values()]):10.1f} '
          f'{np.mean([a["e_psnr"] for a in agg.values()]):9.2f} '
          f'{np.mean([a["e_ssim"] for a in agg.values()]):6.3f} '
          f'{np.mean([a["e_iou"] for a in agg.values()]):6.3f} '
          f'{np.mean([a["e_bright"] for a in agg.values()]):6.3f} '
          f'{np.mean([a["w_psnr"] for a in agg.values()]):8.2f} '
          f'{np.mean([a["w_ssim"] for a in agg.values()]):6.3f} '
          f'{np.mean([a["w_bright"] for a in agg.values()]):6.3f}')
    print("\narm silhouette IoU (exterior camera) and joint write-in error:")
    for seg, a in agg.items():
        print(f'  {seg:14s} arm IoU {a["arm_iou"]:.3f}   qpos write-in error {a["qpos_err"]:.4f} deg')

    out["summary"] = agg
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
