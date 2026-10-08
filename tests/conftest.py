"""Make the package and the dependency-free scripts importable without installing anything."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for sub in ("src", "scripts/eval", "scripts/real2sim"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
