"""Export joint trajectories from a LeRobot v2.1 dataset to npz, one file per episode.

The source is real teleop data (the lab's private 2026-02-09 teleoperation set: 228 episodes on our FR3, 15 Hz,
per-step measured qpos). Reading follows the LeRobot v2.1 on-disk layout: `meta/info.json`
for fps and paths, `meta/episodes.jsonl` and `meta/tasks.jsonl` for the index, and one
parquet per episode under `data/chunk-*/`. Columns verified on this dataset before writing
this script: `joint_position` (7, measured), `gripper_position` (1, closure 0..1),
`actions` (8: the first seven match (qpos[t+1]-qpos[t])*fps to 0.09 rad/s, i.e. commanded
joint *velocities*; the eighth is a continuous gripper command).

The npz is what `replay_rollout_qpos.py --npz` consumes: the measured qpos stream is the
thing to replay through the absolute-joint-position action head, the velocity actions are
carried along only for reference. `--with-images` additionally decodes the two camera
streams stored inline in the parquet (dtype "image", 180x320) into uint8 arrays, for the
real-vs-sim replay videos (`render_rollout_video.py --npz`).

Runs in the polaris openpi venv (pandas + pyarrow live there):
    $POLARIS_ROOT/third_party/openpi/.venv/bin/python \
        scripts/real2sim/export_lerobot_qpos.py --episodes 200 201 202 203 204 120 220 222 \
        --out-dir $WORK/feb_qpos
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("TELEOP_ROOT"), required="TELEOP_ROOT" not in os.environ,
                   help="LeRobot v2.1 dataset directory (the lab's teleoperation set is private)")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--task-contains", default=None,
                   help="alternatively, export every episode whose task contains this")
    p.add_argument("--with-images", action="store_true",
                   help="decode and include both camera streams (for replay videos)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    root = Path(args.root)
    info = json.load(open(root / "meta" / "info.json"))
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(root / "meta" / "tasks.jsonl")}
    episodes = [json.loads(l) for l in open(root / "meta" / "episodes.jsonl")]

    picked = args.episodes or []
    if args.task_contains:
        picked += [e["episode_index"] for e in episodes
                   if any(args.task_contains in t for t in e["tasks"])]
    if not picked:
        raise SystemExit("nothing selected; pass --episodes or --task-contains")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in sorted(set(picked)):
        df = pd.read_parquet(
            root / info["data_path"].format(episode_chunk=idx // info["chunks_size"],
                                            episode_index=idx))
        task = tasks[int(df["task_index"].iloc[0])]
        out = out_dir / f"episode_{idx:06d}.npz"
        arrays = dict(
            qpos=np.stack(df["joint_position"].to_numpy()).astype(np.float64),
            gripper=np.stack(df["gripper_position"].to_numpy()).astype(np.float64),
            actions=np.stack(df["actions"].to_numpy()).astype(np.float64),
            fps=float(info["fps"]),
            task=task,
            episode_index=idx,
        )
        if args.with_images:
            import io

            from PIL import Image
            for col, key in (("exterior_image_1_left", "ext_images"),
                             ("wrist_image_left", "wrist_images")):
                cells = df[col].to_numpy()
                imgs = []
                for c in cells:
                    # LeRobot stores inline images as {'bytes': ..., 'path': None}
                    buf = c["bytes"] if isinstance(c, dict) else c
                    imgs.append(np.asarray(Image.open(io.BytesIO(buf)).convert("RGB")))
                arrays[key] = np.stack(imgs)
        np.savez_compressed(out, **arrays)
        print(f"{out.name}: {len(df)} frames, fps {info['fps']}, task '{task}'"
              + (", with images" if args.with_images else ""))


if __name__ == "__main__":
    main()
