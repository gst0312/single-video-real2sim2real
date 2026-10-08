"""Fix the collision approximation token IsaacLab 2.3.0 writes for meshSimplification.

`isaaclab/sim/schemas/schemas.py` derives the `physics:approximation` value from the
config class name: `TriangleMeshSimplificationPropertiesCfg` becomes
`triangleMeshSimplification`. UsdPhysics has no such token — the valid one is
`meshSimplification`, which is what the official PolaRiS scene assets carry — so PhysX
rejects it at load time with "using unknown value for physics:approximation attribute".

Run this on a converted mesh.usd to write the token the official assets use. Every other
approximation IsaacLab supports round-trips correctly, so this is only needed for
meshSimplification.
"""

import argparse

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
_args_cli, _ = _parser.parse_known_args()
_args_cli.headless = True
_app = AppLauncher(_args_cli).app

from pxr import Usd  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("usd", help="mesh.usd produced by convert_mesh.py")
    p.add_argument("--approximation", default="meshSimplification")
    args = p.parse_args()

    stage = Usd.Stage.Open(args.usd)
    changed = 0
    for prim in stage.Traverse():
        attr = prim.GetAttribute("physics:approximation")
        if attr and attr.HasAuthoredValue() and attr.Get() != args.approximation:
            print(f"{prim.GetPath()}: {attr.Get()!r} -> {args.approximation!r}")
            attr.Set(args.approximation)
            changed += 1
    if changed:
        stage.GetRootLayer().Save()
    print(f"{changed} prim(s) updated in {args.usd}")


if __name__ == "__main__":
    main()
    _app.close()
