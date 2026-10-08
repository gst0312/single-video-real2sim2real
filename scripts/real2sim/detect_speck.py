"""Flag frames whose SIM panel contains the blue-cup + mustard speck pattern.

The leaked raytraced objects render as a small blue chunk with a yellow chunk within a
few pixels; a lone blue or lone yellow region (sky, gripper pads, mustard-coloured cloth)
does not trip it. Runs over every frame of every given real|sim video.
"""

import sys

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage

bad = 0
for path in sys.argv[1:]:
    rd = imageio.get_reader(path)
    hits = []
    for i, frame in enumerate(rd):
        h, w = frame.shape[:2]
        # sim is the second panel; layouts are real|sim or real|sim|blend
        panels = 3 if w / h > 4.5 else 2
        sim = frame[:, w // panels:2 * (w // panels)].astype(int)
        r, g, b = sim[..., 0], sim[..., 1], sim[..., 2]
        blue = (b > r + 25) & (b > g + 18) & (r < 120)
        yellow = (r > 140) & (g > 110) & (b < r - 50)
        if blue.sum() < 3 or yellow.sum() < 3:
            continue
        # a real speck needs a blue cluster and a yellow cluster within 12 px, floating
        # on a flat surround: the leak sat on uniform grey cloth or sky, while colourful
        # scene content (background desk clutter the wrist really sees) is colourful all
        # around and must not trip the gate
        lab_b, nb = ndimage.label(blue)
        lab_y, ny = ndimage.label(yellow)
        yb, xb = np.nonzero(blue)
        found = False
        blue_labels = lab_b[yb, xb]
        blue_sizes = np.bincount(lab_b.ravel())
        for cy in range(1, ny + 1):
            ys, xs = np.nonzero(lab_y == cy)
            if not 3 <= len(ys) <= 80:  # the leak's yellow chunk is ~10 px at panel res
                continue
            d = np.abs(yb[:, None] - ys[None, :]).min(1) + np.abs(xb[:, None] - xs[None, :]).min(1)
            near_clusters = np.unique(blue_labels[d < 12])
            # every adjacent blue cluster must be speck-sized (~35 px), not scene content
            if len(near_clusters) == 0 or blue_sizes[near_clusters].max() > 150:
                continue
            # the leaked pair is the ONLY colour in an otherwise grey panel; real
            # colourful content (background clutter the wrist genuinely sees) fills
            # far more of the frame than the matched pair itself
            colourful = int(((np.abs(r - g) + np.abs(g - b)) > 30).sum())
            pair = int(blue_sizes[near_clusters].sum() + len(ys))
            if colourful > 3 * pair + 50:
                continue
            # and it floats on a BRIGHT open surround (sky or lit cloth); colourful
            # detail tucked against the dark gripper (bracket stickers the real
            # camera also sees) sits on a dark surround and is genuine content
            y0 = max(0, ys.min() - 40)
            y1 = min(sim.shape[0], ys.max() + 40)
            x0 = max(0, xs.min() - 40)
            x1 = min(sim.shape[1], xs.max() + 40)
            patch = sim[y0:y1, x0:x1].mean(axis=2)
            ring = ~(blue[y0:y1, x0:x1] | yellow[y0:y1, x0:x1])
            if np.median(patch[ring]) > 80:
                found = True
                break
        if found:
            hits.append(i)
    rd.close()
    name = path.split("/")[-1]
    if hits:
        bad += 1
        print(f"SPECK {name}: frames {hits[:20]}{'...' if len(hits) > 20 else ''} ({len(hits)} total)")
    else:
        print(f"clean {name}")
sys.exit(1 if bad else 0)
