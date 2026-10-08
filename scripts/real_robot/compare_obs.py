"""Ask the policy the same question twice - same joints, sim pictures vs real pictures.

The real robot executes what it is told (follow error stayed at 0.01-0.06 rad all session),
so a rollout that goes wrong went wrong in what the policy asked for. The policy sees five
things: two images, the joint positions, the gripper reading, and the prompt. Holding the
last three fixed and swapping only the images measures what the sim-to-real visual gap is
worth in radians of commanded motion.

Both observations use the home pose, which is where every episode starts and where sim and
the robot agree to 1e-7 rad, so any difference in the answer comes from the pictures.

    scripts/real_robot/compare_obs.py --port 8000 \
        --sim <sim rollout mp4 or frame> --real <first_frames/*.jpg>

Reports, per query: the first commanded step away from the home pose (what the arm would do
in the next 1/15 s), and the spread over the whole 15-step chunk. Repeating a query shows
the sampling noise floor to compare those numbers against - pi0.5 denoises from noise, so
identical input does not give a bit-identical answer.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# openpi_client lives in the PolaRiS venv; point OPENPI_CLIENT_SITE at its site-packages if
# this is run from another environment
if os.environ.get("OPENPI_CLIENT_SITE"):
    sys.path.insert(0, os.environ["OPENPI_CLIENT_SITE"])

#: Where every episode starts: DROID's reset_joints, matched by the training set to 1e-7 rad.
HOME_QPOS = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])
PROMPT = "pour the mustard into the blue cup"


def load_frame(path):
    """First frame of a video, or an image, as RGB uint8."""
    path = Path(path)
    if path.suffix.lower() in {".mp4", ".mov", ".avi"}:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-frames:v", "1", tmp.name,
                            "-loglevel", "error"], check=True)
            from PIL import Image
            return np.asarray(Image.open(tmp.name).convert("RGB"))
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def split_panels(img):
    """Side-by-side capture -> (exterior, wrist)."""
    half = img.shape[1] // 2
    return img[:, :half], img[:, half:]


def observation(frame_path):
    from openpi_client import image_tools

    exterior, wrist = split_panels(load_frame(frame_path))
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(exterior, 224, 224),
        "observation/wrist_image_left": image_tools.resize_with_pad(wrist, 224, 224),
        "observation/joint_position": HOME_QPOS,
        "observation/gripper_position": np.array([0.0]),
        "prompt": PROMPT,
    }


def describe(name, chunks):
    """First commanded step and chunk spread, averaged over repeats."""
    firsts = np.stack([c[0, :7] - HOME_QPOS for c in chunks])
    print(f"{name}")
    print(f"  first step away from home   {np.abs(firsts).max(axis=1).mean():.4f} rad "
          f"(per joint {np.array2string(np.abs(firsts).mean(axis=0), precision=3)})")
    print(f"  travel over the chunk       {np.mean([np.abs(np.diff(c[:, :7], axis=0)).sum() for c in chunks]):.4f} rad")
    print(f"  gripper command             {np.array2string(np.stack([c[:, 7] for c in chunks]).mean(axis=0), precision=2)}")
    if len(chunks) > 1:
        spread = np.abs(firsts - firsts.mean(axis=0)).max()
        print(f"  jitter over {len(chunks)} repeats     {spread:.4f} rad  <- sampling noise floor")
    return firsts.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    a = ap.parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)

    out = {}
    for name, path in (("sim", a.sim), ("real", a.real)):
        obs = observation(path)
        chunks = [np.asarray(client.infer(obs)["actions"]) for _ in range(a.repeats)]
        out[name] = describe(f"\n{name}  ({Path(path).name})", chunks)

    gap = np.abs(out["sim"] - out["real"])
    print(f"\nsame qpos, only the images differ: first commanded step differs by {gap.max():.4f} rad "
          f"(per joint {np.array2string(gap, precision=3)})")


if __name__ == "__main__":
    main()
