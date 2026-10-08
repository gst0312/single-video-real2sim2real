"""The pour criterion, written the way PolaRiS writes criteria.

PolaRiS ships three checkers - `reach`, `lift`, `is_within_xy`
(`polaris/environments/rubrics/checkers.py`) - and composes them into a `Rubric` with
per-criterion dependencies at registration time. Pouring needs one they do not have, so it
is written here in the same closure-factory shape and composed the same way; everything
else about scoring (max-ever-reached per criterion, progress = fraction reached, success =
all reached) stays theirs.

Two kinds of number go into `pouring`, kept apart on purpose:

Geometry comes from the assets at run time, never hardcoded. The bottle's mouth is the top
of its own bounding box along its local +Y (both task meshes stand with +Y up, which is
also what `synthesize_trajectories.detect_quat_order` relies on), and the cup's rim and
radius likewise. The bounding box is read with PolaRiS's own `checkers.get_bbox`, at
identity pose so the numbers come out in the object's local frame. This keeps the criterion
correct if the meshes are ever re-converted with a different origin.

Thresholds come from the human demonstration, measured on `take4.npy` (the tracked object
trajectory, 184 frames, the same file the generator retargets):

- tilt away from upright at the pour: 83-84 degrees, held flat over the last 20 frames.
  The gate is 60 degrees, well clear of both upright and the demo's plateau.
- mouth to cup-centre distance in xy at the pour: 0.019-0.022 m, against a cup radius of
  0.047 m. The gate is the cup's own radius, so "the stream would land in the cup".
- mouth height above the cup rim at the pour: 0.063-0.070 m. The gate is 0, i.e. the mouth
  has to be above the rim rather than beside the cup at rim height, with a ceiling of
  0.30 m so a bottle waved high overhead does not count.
- dwell: the demonstration satisfies mouth-inside-radius and tilt together for 86 of its
  184 frames. The gate is 15 consecutive steps, one second at 15 Hz, which is a small
  fraction of that and rules out a transient sweep across the cup on the way somewhere.
"""

import numpy as np
import torch

from polaris.environments.rubrics import Rubric
from polaris.environments.rubrics import checkers

from r2s2r.pour_geometry import pour_condition, pour_measurements, rotate

IDENTITY_POS = torch.zeros(3)
IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _local_extents(env, name):
    """(min, max) corner of an object's bounding box in its own frame, metres."""
    from omni.usd import get_context

    stage = get_context().get_stage()
    prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")
    corners, _ = checkers.get_bbox(prim, pos=IDENTITY_POS, quat=IDENTITY_QUAT)
    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)


def reach_centre(name, threshold=0.2):
    """PolaRiS's `reach`, measured to the object's centre instead of its origin.

    `checkers.reach` compares the end-effector to `data.root_pos_w`, which is wherever the
    mesh's origin happens to sit. In the official environments that is near the object's
    middle, so 0.2 m means "the gripper is at the object". Our meshes come from GSWorld and
    stand on their origin - the mustard bottle's is at its base, 0.083 m below its centre -
    and the same 0.2 m then means something stricter. Measured on the first physics rollout:
    the closest the gripper base ever came to the bottle's origin was 0.221 m, while it was
    holding the bottle and lifting it 16 cm, so the criterion never fired and, being the
    first in the chain, blocked the two that matter.

    Rather than inflate the threshold - which would make the number mean nothing - the
    origin convention is removed: the comparison point is the centre of the object's own
    bounding box, so PolaRiS's 0.2 m keeps its meaning. Everything else is their checker.
    """
    offset = {}

    def checker(env):
        if name not in offset:
            lo, hi = _local_extents(env, name)
            offset[name] = (lo + hi) / 2
        obj = env.scene[name]
        pos = obj.data.root_pos_w[0].detach().cpu().numpy()
        quat = obj.data.root_quat_w[0].detach().cpu().numpy()
        centre = pos + rotate(quat, offset[name])
        ee = env.scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
        return float(np.linalg.norm(centre - ee)) < threshold

    return checker


def pouring(bottle, cup, tilt_deg=60.0, dwell=15, rim_clearance=(0.0, 0.30),
            radius_scale=1.0):
    """The bottle is tilted over the cup's mouth, and stays there.

    Returns a checker closure with a `reset()` for `PourRubric` to call between episodes,
    since the dwell counter is state the base `Rubric` knows nothing about.
    """
    state = {"held": 0, "geom": None, "best": {}, "trace": []}

    def checker(env):
        if state["geom"] is None:
            b_lo, b_hi = _local_extents(env, bottle)
            c_lo, c_hi = _local_extents(env, cup)
            # +Y is up in both meshes; the cup's radius is the smaller horizontal
            # half-extent so the handle, which sticks out along one axis only, is excluded
            state["geom"] = {
                "mouth": float(b_hi[1]),
                "rim": float(c_hi[1]),
                "radius": float(min(c_hi[0] - c_lo[0], c_hi[2] - c_lo[2]) / 2),
            }
            print(f"[r2s2r] pour rubric geometry from the assets: "
                  f"bottle mouth {state['geom']['mouth']:.4f} m up its own axis, "
                  f"cup rim {state['geom']['rim']:.4f} m, "
                  f"cup radius {state['geom']['radius']:.4f} m")
        g = state["geom"]

        b_pos = env.scene[bottle].data.root_pos_w[0].detach().cpu().numpy()
        b_quat = env.scene[bottle].data.root_quat_w[0].detach().cpu().numpy()
        c_pos = env.scene[cup].data.root_pos_w[0].detach().cpu().numpy()

        # the geometry is `pour_geometry`, kept simulator-free so it can be tested
        tilt, d_xy, above = pour_measurements(b_pos, b_quat, c_pos, g["mouth"], g["rim"])
        now = pour_condition(tilt, d_xy, above, g["radius"], tilt_deg, rim_clearance,
                             radius_scale)
        state["held"] = state["held"] + 1 if now else 0

        # keep the closest approach to each condition, so a failing episode says which one
        # it failed on rather than just "no"
        b = state["best"]
        b["tilt_max_deg"] = max(b.get("tilt_max_deg", -1e9), float(tilt))
        b["mouth_cup_xy_min"] = min(b.get("mouth_cup_xy_min", 1e9), d_xy)
        b["mouth_above_rim_at_closest"] = (
            float(above) if d_xy <= b["mouth_cup_xy_min"] else b.get("mouth_above_rim_at_closest"))
        b["held_max"] = max(b.get("held_max", 0), state["held"])
        b["cup_radius"] = g["radius"]
        state["trace"].append((float(tilt), d_xy, float(above)))
        return state["held"] >= dwell

    def _reset():
        state.update(held=0, best={}, trace=[])

    checker.reset = _reset
    checker.stats = lambda: dict(state["best"])
    #: per-step (tilt_deg, mouth-to-cup xy, mouth above rim) from the first step the
    #: criterion was evaluated, i.e. once `lift` had been reached. A failing episode is
    #: read off this: which of the three is out, and when.
    checker.trace = lambda: list(state["trace"])
    return checker


class PourRubric(Rubric):
    """PolaRiS's Rubric, with the stateful criteria reset too.

    The base class resets only `criteria_reached`; `pouring` carries a dwell counter, and
    its docstring says stateful rubrics should override `reset`.
    """

    def reset(self):
        super().reset()
        for c in self.criteria:
            fn = c[0] if isinstance(c, tuple) else c
            if hasattr(fn, "reset"):
                fn.reset()


def pour_mustard_rubric():
    """The task: pick up the mustard bottle and pour it into the blue cup.

    Same shape as the official environments' rubrics (reach, then lift gated on reach,
    then the task criterion gated on lift). `lift` is PolaRiS's own and `reach_centre` is
    theirs with the origin convention taken out; the thresholds are theirs too (reach
    0.2 m; lift 0.06 m, against a demonstration that raises the bottle 0.165 m).
    """
    return PourRubric(criteria=[
        reach_centre("mustard", threshold=0.2),
        (checkers.lift("mustard", threshold=0.06), [0]),
        (pouring("mustard", "blue_cup"), [1]),
    ])
