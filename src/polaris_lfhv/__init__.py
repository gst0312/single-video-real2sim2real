"""Single human video to robot data, on top of PolaRiS and Real2Render2Real.

Only the pieces that do not exist upstream live here. The environment, action space,
renderer, evaluation loop and training configs are used from polaris and openpi as they
are; see docs/provenance.md for what came from where.

Importable without a simulator: `limits` (FR3 joint limits, velocity envelope, motion
budget), `physics_gate` (which replayed episodes become data), `pour_geometry`
(the pour criterion's maths), `alignment` and `robot_links`. Everything under
`environments` that touches Isaac, and `training`, needs the PolaRiS / openpi stacks.
"""
