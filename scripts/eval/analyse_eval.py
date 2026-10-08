"""Join evaluation outcomes with the layout each episode was run on.

A success rate on its own says how good the policy is but not what to do next. The
conditions differ only in where the two objects sit, so pairing each outcome with its
layout says whether the failures are spread evenly - in which case the answer is more
training - or clustered in one corner of the workspace, in which case the answer is more
data there.

    python scripts/eval/analyse_eval.py <eval_results.csv> <conditions.json>

The episode column indexes into the conditions file, which is what shard_conditions.py's
merge restores after a sharded run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(results, conditions):
    rows = list(csv.DictReader(Path(results).open()))
    poses = json.loads(Path(conditions).read_text())["poses"]
    out = []
    for r in rows:
        i = int(r["episode"])
        if i >= len(poses):
            continue
        p = poses[i]
        out.append({
            "i": i,
            "ok": r["success"] == "True",
            "progress": float(r["progress"]),
            "steps": int(r["episode_length"]),
            "bottle": np.array(p["mustard"][:2]),
            "cup": np.array(p["blue_cup"][:2]),
        })
    return out


def report(rows):
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    n = len(rows)
    print(f"{len(ok)}/{n} succeeded ({100 * len(ok) / n:.0f}%)")

    # progress tells apart "never grasped" from "grasped but did not pour": the rubric's
    # three criteria are reach, lift, pour, so 2/3 means everything but the pour.
    from collections import Counter
    dist = Counter(round(r["progress"], 3) for r in rows)
    print("progress distribution: " + ", ".join(f"{k}: {v} episodes" for k, v in sorted(dist.items())))
    if bad:
        stuck = sum(1 for r in bad if r["progress"] >= 0.66)
        print(f"  of the {len(bad)} failures, {stuck} grasped but did not pour, "
              f"{len(bad) - stuck} never completed the grasp")

    print(f"steps: median over successes {np.median([r['steps'] for r in ok]):.0f}" if ok
          else "steps: no successful episodes")

    for name in ("bottle", "cup"):
        print(f"\n{name} position (m):")
        for label, group in (("success", ok), ("failure", bad)):
            if not group:
                continue
            a = np.array([r[name] for r in group])
            print(f"  {label} n={len(a):2d}  x mean {a[:,0].mean():.3f} range [{a[:,0].min():.3f},{a[:,0].max():.3f}]"
                  f"  y mean {a[:,1].mean():.3f} range [{a[:,1].min():.3f},{a[:,1].max():.3f}]")

    # the layout parameter that actually varies the task is where the cup sits relative to
    # the bottle - that vector is the carry the policy has to perform
    print("\ncarry vector (cup - bottle):")
    for label, group in (("success", ok), ("failure", bad)):
        if not group:
            continue
        d = np.array([r["cup"] - r["bottle"] for r in group])
        dist_m = np.linalg.norm(d, axis=1)
        print(f"  {label} n={len(d):2d}  distance mean {dist_m.mean():.3f} range [{dist_m.min():.3f},{dist_m.max():.3f}]")

    if ok and bad:
        db = np.linalg.norm(np.array([r["cup"] - r["bottle"] for r in bad]), axis=1)
        dg = np.linalg.norm(np.array([r["cup"] - r["bottle"] for r in ok]), axis=1)
        gap = db.mean() - dg.mean()
        print(f"  failures carry {abs(gap) * 100:.1f} cm {'farther' if gap > 0 else 'shorter'} than successes")

    print("\nper episode (condition index: ok, or the progress reached on failure):")
    cells = []
    for r in rows:
        cells.append(f"{r['i']}:ok" if r["ok"] else f"{r['i']}:{r['progress']:.2f}")
    print("  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("conditions")
    a = ap.parse_args()
    report(load(a.results, a.conditions))


if __name__ == "__main__":
    main()
