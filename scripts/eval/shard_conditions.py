"""Split an initial-conditions file into shards so evaluation can run on several GPUs.

PolaRiS's `eval.py` walks one conditions file from the top and takes `--rollouts` of them,
so the only way to spread 30 conditions over N processes without touching their script is to
hand each process its own file. The shards are contiguous slices, not a round robin, so a
shard's episode numbering stays readable against the original file: shard 0 is conditions
0..k, shard 1 is k..2k, and `merge` puts the original indices back.

    python scripts/eval/shard_conditions.py split <conditions.json> <out_dir> --shards 3
    python scripts/eval/shard_conditions.py merge <out_dir> --shards 3
"""

import argparse
import csv
import json
from pathlib import Path


def split(conditions, out_dir, shards):
    src = json.loads(Path(conditions).read_text())
    poses = src["poses"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ceil division, so the last shard is the short one rather than a whole extra shard
    per = -(-len(poses) // shards)
    for i in range(shards):
        part = poses[i * per : (i + 1) * per]
        if not part:
            continue
        path = out_dir / f"shard{i}.json"
        path.write_text(json.dumps({**src, "poses": part}, indent=2))
        print(f"{path}: {len(part)} conditions (original index {i * per}..{i * per + len(part) - 1})")
    return per


def merge(out_dir, shards, per):
    """Concatenate the shard CSVs, restoring the original condition index."""
    out_dir = Path(out_dir)
    rows = []
    for i in range(shards):
        csv_path = out_dir / f"shard{i}" / "eval_results.csv"
        if not csv_path.exists():
            print(f"missing {csv_path}")
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                row["episode"] = str(i * per + int(row["episode"]))
                rows.append(row)
    rows.sort(key=lambda r: int(r["episode"]))
    if not rows:
        return
    merged = out_dir / "eval_results.csv"
    with merged.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["success"] == "True" for r in rows)
    mean_progress = sum(float(r["progress"]) for r in rows) / len(rows)
    print(f"{merged}: {n_ok}/{len(rows)} success ({100 * n_ok / len(rows):.0f}%), "
          f"mean progress {mean_progress:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "merge"])
    ap.add_argument("path")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--per", type=int, default=0, help="merge: conditions per shard")
    a = ap.parse_args()
    if a.mode == "split":
        split(a.path, a.out_dir, a.shards)
    else:
        merge(a.path, a.shards, a.per)


if __name__ == "__main__":
    main()
