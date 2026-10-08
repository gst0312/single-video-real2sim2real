"""The physics gate: which replayed episodes become training data.

`scripts/datagen/replay_reference_episode.py` writes a report json for every replayed
episode, failures included. The gate reads that report and keeps an episode only if the
arm actually grasped the bottle, carried it along the reference, and did not knock the
cup. Whether the pour landed in the cup is deliberately NOT a criterion: the rigid-follow
assumption behind the synthesised trajectories does not model contact compliance or
liquid, so it is not a fair reason to drop data (user decision, 2026-08-13). The rubric
still scores every episode; the gate is about executability, the rubric about the task.

Shared by `scripts/datagen/build_rlds.py` (what gets written) and
`scripts/datagen/analyse_dataset.py` (what gets counted), so the two can never disagree.
"""

LIFT_MIN = 0.06   # m: the bottle has to come up this far, i.e. it was grasped; same
                  # threshold as the rubric's lift criterion
GAP_MAX = 0.05    # m: how far the bottle may drift from the reference before it counts as
                  # knocked askew rather than carried
CUP_MAX = 0.02    # m: cup displacement that counts as a collision


def physics_gate_reasons(report, lift_min=LIFT_MIN, gap_max=GAP_MAX, cup_max=CUP_MAX):
    """Why a replayed episode fails the physics gate; an empty list means it passes.

    `report` is the dict `replay_reference_episode.py` writes: `bottle_lift`,
    `bottle_vs_reference.gap_max_m` and `cup_moved_m` are the fields read. A missing
    `gap_max_m` (the replay never reached the grasp step) counts as not carried.
    """
    why = []
    lift = float(report["bottle_lift"])
    if lift < lift_min:
        why.append(f"lift {lift * 100:.1f} cm")
    gap = report["bottle_vs_reference"]["gap_max_m"]
    if gap is None:
        why.append("never reached the grasp step")
    elif float(gap) > gap_max:
        why.append(f"drifted {float(gap) * 100:.1f} cm from the reference")
    cup = float(report["cup_moved_m"])
    if cup > cup_max:
        why.append(f"cup moved {cup * 100:.1f} cm")
    return why


def passes_physics_gate(report, lift_min=LIFT_MIN, gap_max=GAP_MAX, cup_max=CUP_MAX):
    return not physics_gate_reasons(report, lift_min, gap_max, cup_max)
