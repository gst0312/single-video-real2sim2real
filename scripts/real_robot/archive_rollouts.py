#!/usr/bin/env python3
"""Copy new real-robot rollouts into this repo, compressed enough to browse on GitHub.

The eval script writes each episode as a 20-30 MB mp4 next to the run's first-frame png,
under a directory per launch. Those stay where they are -- this only ever reads them --
and a 960-wide re-encode plus the first frame lands in `docs/real_robot/<session>/`.

Run it after an episode (or a batch of them); it is idempotent, so re-running only picks
up what is new. Episode numbers come from the recording timestamp, so they stay stable as
more episodes arrive.

    python scripts/real_robot/archive_rollouts.py                 # today, default paths
    python scripts/real_robot/archive_rollouts.py --date 20260814
    python scripts/real_robot/archive_rollouts.py --rollouts $ROLLOUTS --dry-run

Scores are read from the eval csv (a separate tree -- the eval script writes the video and
the csv to different roots) and merged into `episodes.csv`. Note the csv can disagree with
what actually happened: before the launcher's scoring patch, a success typed as `y` was
recorded as 0.01. The session README is where the human-verified outcome lives.
"""

import argparse
import csv
import os
import datetime
import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# where the DROID eval client writes videos and csvs on the robot laptop; lab specific,
# so both come from the environment
DEFAULT_ROLLOUTS = os.environ.get("ROLLOUTS", "rollouts")
DEFAULT_EVAL_CSVS = os.environ.get("EVAL_CSVS", "eval_csvs")
VIDEO_WIDTH = 960
VIDEO_CRF = 30


def ffmpeg_exe():
    for candidate in (shutil.which("ffmpeg"),):
        if candidate:
            return candidate
    try:  # the DROID venv ships one with imageio
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("no ffmpeg: install it, or run this with the DROID venv's python")


def session_dir(date):
    """Existing docs/real_robot/<date>* if there is exactly one, else create <date>."""
    root = os.path.join(REPO, "docs", "real_robot")
    matches = sorted(glob.glob(os.path.join(root, date + "*")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sys.exit("ambiguous session directories for %s:\n  %s" % (date, "\n  ".join(matches)))
    return os.path.join(root, date)


def episode_stamp(path):
    """The leading timestamp of a rollout filename: 20260814021236-pi05-pour ... .mp4"""
    return os.path.basename(path).split("-", 1)[0]


def read_scores(eval_csv_root, date):
    """{video basename without extension: success} from the eval script's csv files."""
    scores = {}
    for path in sorted(glob.glob(os.path.join(eval_csv_root, date + "*", "*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                name = os.path.basename(row.get("video_filename", ""))
                if name:
                    scores[name] = row.get("success", "")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS,
                        help="where the eval script writes videos (default %s)" % DEFAULT_ROLLOUTS)
    parser.add_argument("--eval-csvs", default=DEFAULT_EVAL_CSVS,
                        help="where it writes the score csv (default %s)" % DEFAULT_EVAL_CSVS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.rollouts, args.date + "*", "*.mp4")),
                    key=episode_stamp)
    if not videos:
        print("no rollouts for %s under %s" % (args.date, args.rollouts))
        return 0

    out = session_dir(args.date)
    scores = read_scores(args.eval_csvs, args.date)
    ffmpeg = ffmpeg_exe()
    added = []

    for index, source in enumerate(videos, start=1):
        name = "ep%02d_%s.mp4" % (index, episode_stamp(source))
        target = os.path.join(out, "videos", name)
        if os.path.exists(target):
            continue
        added.append((name, source))
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", source,
             "-vf", "scale=%d:-2" % VIDEO_WIDTH, "-c:v", "libx264",
             "-crf", str(VIDEO_CRF), "-preset", "veryfast", "-an", target],
            check=True,
        )

    # First frames: one per launch, not per episode.
    for first in sorted(glob.glob(os.path.join(args.rollouts, args.date + "*",
                                               "robot_camera_views.png"))):
        run = os.path.basename(os.path.dirname(first))
        target = os.path.join(out, "first_frames", run + ".jpg")
        if os.path.exists(target) or args.dry_run:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        from PIL import Image

        image = Image.open(first)
        image.resize((image.width // 2, image.height // 2)).convert("RGB").save(target, quality=82)

    # The eval script's own csv files, copied verbatim. episodes.csv below is the index;
    # these are the originals, kept so the archive does not depend on a tree outside it.
    for source in sorted(glob.glob(os.path.join(args.eval_csvs, args.date + "*", "*.csv"))):
        run = os.path.basename(os.path.dirname(source))
        target = os.path.join(out, "eval_csv", "%s_%s" % (run, os.path.basename(source)))
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)  # re-copied every run: a rerun can append rows

    if not args.dry_run:
        rows = [("episode", "recorded", "run", "success_from_csv", "source_video")]
        for index, source in enumerate(videos, start=1):
            stem = os.path.splitext(os.path.basename(source))[0]
            rows.append(("ep%02d" % index, episode_stamp(source),
                         os.path.basename(os.path.dirname(source)),
                         scores.get(stem, ""), source))
        with open(os.path.join(out, "episodes.csv"), "w", newline="") as handle:
            csv.writer(handle).writerows(rows)

    print("%s: %d episode(s) total, %d newly archived -> %s"
          % (args.date, len(videos), len(added), out))
    for name, source in added:
        print("  + %s  <- %s" % (name, source))
    if added and not args.dry_run:
        print("\nthe session README is not written for you: add the human-verified outcome "
              "for the new episodes, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
