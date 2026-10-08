"""Tile the single-frame check the iteration loop reports with: real | GSWorld | ours | blend.

GSWorld publishes per-frame real|sim pairs for four segments
(`GSWorld/docs/renders/scene06/frames/<seg>_<cam>_f<idx>_real_sim.png`); our
`compare_to_real_rollout.py` writes `_real.png` / `_sim.png` for the same frames. This
script only concatenates: column 1 the real frame, column 2 GSWorld's own render of the
same frame (the right panel of their pair, their quality bar), column 3 ours, column 4 a
50/50 blend of ours over real (same blend `compare_to_real_rollout.py` writes). The
mechanism is GSWorld's own frames_10x contact-sheet concatenation, one row, with labels.

    python scripts/real2sim/make_four_column.py \
        --real  out/randomwalk_2_000660_external_cam_real.png \
        --gsworld $GSWORLD_ROOT/docs/renders/scene06/frames/randomwalk_2_third_f000660_real_sim.png \
        --ours  out/randomwalk_2_000660_external_cam_sim.png \
        --out   out/randomwalk_2_000660_external_cam_4col.jpg
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw


def load(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


def label(img, text):
    draw = ImageDraw.Draw(img)
    # default bitmap font is tiny at 720p; draw it scaled up via a small canvas instead
    patch = Image.new("RGB", (len(text) * 7 + 6, 14), (0, 0, 0))
    ImageDraw.Draw(patch).text((3, 1), text, fill=(255, 255, 255))
    patch = patch.resize((patch.width * 3, patch.height * 3), Image.NEAREST)
    img.paste(patch, (12, 12))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--gsworld", help="GSWorld's published pair for the same frame; "
                                     "omit the column if they never rendered it")
    p.add_argument("--gsworld-panel", default="right", choices=["left", "right", "full"],
                   help="their pairs are real|sim, so their render is the right half")
    p.add_argument("--ours", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ours = Image.open(args.ours).convert("RGB")
    size = ours.size
    real = load(args.real, size)

    columns = [label(real.copy(), "real"), ]
    if args.gsworld:
        gs = Image.open(args.gsworld).convert("RGB")
        if args.gsworld_panel != "full":
            half = gs.width // 2
            box = (0, 0, half, gs.height) if args.gsworld_panel == "left" \
                else (gs.width - half, 0, gs.width, gs.height)
            gs = gs.crop(box)
        if gs.size != size:
            gs = gs.resize(size, Image.LANCZOS)
        columns.append(label(gs, "GSWorld"))
    blend = Image.fromarray(((np.asarray(real, np.float32)
                              + np.asarray(ours, np.float32)) / 2).astype(np.uint8))
    columns += [label(ours.copy(), "ours"), label(blend, "ours + real")]

    strip = Image.new("RGB", (size[0] * len(columns) + 4 * (len(columns) - 1), size[1]),
                      (255, 255, 255))
    for i, col in enumerate(columns):
        strip.paste(col, (i * (size[0] + 4), 0))
    strip.save(args.out, quality=92)
    print(f"wrote {args.out} ({len(columns)} columns)")


if __name__ == "__main__":
    main()
