"""Geometry of the pour criterion, free of any simulator dependency.

`environments/rubrics.py` wraps these in a PolaRiS checker closure; the maths lives here so it can be
unit tested and exercised on a synthetic object track without Isaac. Conventions: positions
are metres in the world frame, quaternions are (w, x, y, z), both task meshes stand with
their long axis along their local +Y, and the bottle's mouth is at `mouth` metres up that
axis while the cup's rim is `rim` metres above the cup's origin.
"""

import numpy as np


def rotate(quat_wxyz, v):
    """Rotate v by a wxyz quaternion, without pulling in a rotation library."""
    w, x, y, z = quat_wxyz
    u = np.array([x, y, z], dtype=float)
    v = np.asarray(v, dtype=float)
    return 2 * u.dot(v) * u + (w * w - u.dot(u)) * v + 2 * w * np.cross(u, v)


def pour_measurements(bottle_pos, bottle_quat_wxyz, cup_pos, mouth, rim):
    """(tilt_deg, mouth_to_cup_xy, mouth_above_rim) for one step.

    tilt is the angle between the bottle's up axis and world +Z (0 upright, 90 horizontal);
    mouth_to_cup_xy is the horizontal distance from the bottle's mouth to the cup's axis;
    mouth_above_rim is the mouth's height above the cup's rim (negative: below it).
    """
    axis = rotate(bottle_quat_wxyz, np.array([0.0, 1.0, 0.0]))
    tilt = float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0))))
    mouth_pos = np.asarray(bottle_pos, dtype=float) + axis * mouth
    cup_pos = np.asarray(cup_pos, dtype=float)
    d_xy = float(np.linalg.norm(mouth_pos[:2] - cup_pos[:2]))
    above = float(mouth_pos[2] - (cup_pos[2] + rim))
    return tilt, d_xy, above


def pour_condition(tilt, d_xy, above, radius, tilt_deg=60.0, rim_clearance=(0.0, 0.30),
                   radius_scale=1.0):
    """Is the bottle tilted over the cup's mouth at this step?

    tilt past `tilt_deg`, mouth within `radius * radius_scale` of the cup's axis, and the
    mouth above the rim but not more than `rim_clearance[1]` above it (a bottle waved high
    overhead does not count).
    """
    return bool(tilt > tilt_deg and d_xy < radius * radius_scale
                and rim_clearance[0] < above < rim_clearance[1])


class DwellCounter:
    """Counts consecutive steps a condition holds; the pour must be held, not swept through."""

    def __init__(self, dwell):
        self.dwell = int(dwell)
        self.held = 0
        self.best = 0

    def update(self, now):
        """Feed one step's condition; returns True once it has held for `dwell` steps."""
        self.held = self.held + 1 if now else 0
        self.best = max(self.best, self.held)
        return self.held >= self.dwell

    def reset(self):
        self.held = 0
        self.best = 0
