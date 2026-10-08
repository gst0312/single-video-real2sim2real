"""Say whether the objects are placed where the policy was trained to see them.

The training layouts live in a 10 x 9 cm box for the bottle and 6 x 6 cm for the cup, in
the robot's base frame - numbers that are awkward to measure on a table with a ruler. The
wrist camera gives the same information for free: at the home pose it looks straight at the
bottle, so how large the bottle appears is a direct read on how far away it is, and apparent
area goes as 1/distance^2.

Calibrated against the 30 held-out sim episodes the final checkpoint scored 30/30 on, and
checked against the first real session (2026-08-14): every failed episode had the bottle
0.95-1.06% of the wrist frame against sim's 1.31% median - roughly 12% too far, about 4 cm -
and all four that succeeded measured 1.14-1.37%. Nothing lands in between, so the check is
a threshold at 1.10 and it gets all 13 episodes of that session right.

    python scripts/real_robot/check_placement.py <image>

Takes the launcher's first-frame capture, `robot_camera_views.png` from a rollout directory,
or a dual-view rollout mp4 (its first frame is the same picture): exterior on the left, wrist
on the right. A sim rollout frame works too - the letterbox resize_with_pad leaves is detected
and cropped.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Bottle appearance in the wrist frame across the 30 sim eval episodes: median and the
#: full range. Fractions are of the wrist image's content area, percent.
SIM_AREA = (1.309, 0.833, 1.499)
SIM_CX = (0.399, 0.318, 0.447)
SIM_CY = (0.517, 0.438, 0.609)

#: Accept a band around the sim median rather than its full range. The floor is set from the
#: real session, where the two groups separate cleanly with nothing in between: the largest
#: any failure measured was 1.06% (ep10) and the smallest any success measured was 1.14%
#: (ep11), so 1.10 splits them with margin on both sides and gets all 13 episodes right.
AREA_OK = (1.10, 1.55)


def load(path):
    """An image, or the first frame of a rollout video, as RGB.

    The launcher only saved a separate first-frame capture for part of the session, but a
    dual-view recording carries the same thing in its first frame, so an mp4 works too and
    the archived videos can be checked directly.
    """
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "f.png"
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-frames:v", "1", str(frame),
                            "-loglevel", "error"], check=True)
            return np.asarray(Image.open(frame).convert("RGB")).astype(float)
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def yellow(arr):
    """Mask of the mustard bottle: strongly yellow pixels."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r > 110) & (g > 90) & (b < 0.55 * r) & ((r + g) / 2 - b > 45)


#: Width over height below which the picture cannot be two views side by side. A dual-view
#: capture is about 3.6:1 (1280x360, 960x270); a single camera is 16:9. Without this check a
#: single-view recording silently gets its right half measured as if it were the wrist, which
#: reads as a tiny bottle and answers "too far" with full confidence.
MIN_ASPECT = 2.5


def wrist_panel(img):
    """Right half of a side-by-side capture, with any letterbox rows removed."""
    h, w = img.shape[:2]
    if w / h < MIN_ASPECT:
        raise SystemExit(
            f"this image is {w}x{h} (aspect {w / h:.2f}), not a side-by-side dual view.\n"
            "Pass a picture with the exterior and wrist cameras side by side: the launcher's\n"
            "first-frame capture, robot_camera_views.png from a rollout directory, or a video\n"
            "recorded with --record-wrist. A single-view recording has no wrist view, so the\n"
            "bottle-to-gripper distance cannot be measured."
        )
    panel = img[:, w // 2:]
    lit = np.nonzero(panel.max(axis=(1, 2)) > 12)[0]
    if len(lit) and (lit.max() - lit.min() + 1) < 0.95 * panel.shape[0]:
        panel = panel[lit.min(): lit.max() + 1]     # sim frames are padded to square
    return panel


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-3].strip())
    panel = wrist_panel(load(sys.argv[1]))
    mask = yellow(panel)
    if mask.sum() < 30:
        raise SystemExit("mustard bottle not found: check that the right half is the wrist camera "
                         "and that the bottle is in view")

    h, w = panel.shape[:2]
    ys, xs = np.nonzero(mask)
    area = mask.sum() / (h * w) * 100
    cx, cy = xs.mean() / w, ys.mean() / h

    print(f"wrist view {w}x{h}")
    print(f"  bottle area   {area:.2f}%    sim median {SIM_AREA[0]:.2f}"
          f" (range over 30 episodes {SIM_AREA[1]:.2f}-{SIM_AREA[2]:.2f})")
    print(f"  centroid x    {cx:.3f}     sim median {SIM_CX[0]:.3f}")
    print(f"  centroid y    {cy:.3f}     sim median {SIM_CY[0]:.3f}")
    print()

    if area < AREA_OK[0]:
        # area ~ 1/d^2, so the distance ratio is sqrt(sim / measured)
        far = (np.sqrt(SIM_AREA[0] / area) - 1) * 100
        print(f"bottle too far: about {far:.0f}% farther than the typical training placement "
              f"(roughly {0.30 * far / 100 * 100:.0f} cm at 30 cm)")
        print("Move the bottle towards the gripper until the area reads 1.2-1.4%, then start.")
        print("In the 2026-08-14 session every failure measured 0.95-1.06%; the first success was 1.27%.")
        return 1
    if area > AREA_OK[1]:
        print(f"bottle too close (area {area:.2f}% above {AREA_OK[1]}%); move it out a little.")
        return 1
    print(f"distance ok (area within {AREA_OK[0]}-{AREA_OK[1]}%); clear to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
