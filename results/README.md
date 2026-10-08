# results

Small logs and summaries that back the numbers in `docs/results.md`, plus a few frames.
The demonstration video is committed (`videos/demo_sim_to_real.mp4`, 5.3 MB, simulation rollout followed by the zero-shot real-robot run at 3× speed) together with its GIF (`figures/demo_sim_to_real.gif`). Everything else large (the per-episode comparison videos, the RLDS dataset, the checkpoint, scene assets) stays out of the
repository.

- `phase5_eval/` — checkpoint sweep on 30 held-out placements: one CSV per checkpoint
  (`sweep/step*.csv`, columns `episode, episode_length, success, progress`) and the base
  policy's `base_zero_shot.csv`. Episode index = condition index in `eval_conditions.json`.
- `phase3_rollouts/` — the first three physics replays of synthesised trajectories (the
  "can the stock gripper hold this bottle" smoke test), full per-episode reports as written
  by `replay_reference_episode.py`.
- `replay_validation/` — scene and action-convention validation: `rollout_frames_summary.json`
  (PSNR/SSIM/IoU/brightness per frame and per segment against eight real recordings),
  `replay_gsworld_recordings/` and `replay_teleop/` (tracking error of recorded joint angles
  replayed as absolute-position actions).
- `real_robot_20260814/` — the first real-robot session: `episodes.csv` (recording and run
  ids, the raw score from the evaluation CSV, which recorded successes as 0.01), the
  human-verified outcome table in its README, and two first-frame captures. The session's
  demonstration video is public: [Demo video (Google Drive)](https://drive.google.com/file/d/1L6XNSg5zBRMjBuRxUfQErBS44NhcPLfO/view?usp=drive_link).
- `trajectory_synthesis.md` — per-condition gate table of the first synthesis round.
- `figures/` — eight frames: the demonstration beside the reconstructed scene, the
  environment's two views at a reset, three real-vs-sim frames at recorded joint angles, the
  real / GSWorld / ours / blend comparison, and the dataset coverage plot.

## Files for a GitHub Release

These were produced and viewed during the project and are listed so they can be attached to a
Release rather than committed. Sizes are of the files as produced.

Real-vs-sim videos, arm posed at the recorded joint angles, three columns (real | sim | 50%
blend), marker-free environment:

| file | size |
|---|---|
| `home_static_{external,wrist}_cam_real_sim.mp4` | 3.2 + 4.1 MB |
| `randomwalk_1_{external,wrist}_cam_real_sim.mp4` | 13.9 + 19.8 MB |
| `randomwalk_2_{external,wrist}_cam_real_sim.mp4` | 15.0 + 22.3 MB |
| `randomwalk_3_{external,wrist}_cam_real_sim.mp4` | 15.7 + 24.5 MB |
| `randomwalk_4_{external,wrist}_cam_real_sim.mp4` | 15.3 + 21.7 MB |
| `vertical_1_{external,wrist}_cam_real_sim.mp4` | 4.5 + 5.8 MB |
| `wristroll_1_{external,wrist}_cam_real_sim.mp4` | 8.8 + 13.7 MB |
| `wristroll_2_{external,wrist}_cam_real_sim.mp4` | 9.2 + 13.5 MB |

The same sixteen for the marker variant (`with_aruco`, 54 MB in total, rendered on an
earlier asset revision).

Teleoperation replays (real frames from the dataset beside the simulator at the measured
joint angles; the objects differ by design): `hf_000200_*`, `hf_000203_*`, `hf_000220_*`,
external and wrist, 1.1-14.2 MB each, 28 MB in total.

Pipeline clips: phase-3 physics replays `cond000_ep03/05/06.mp4` (about 2 MB each, exterior
and wrist side by side, 2x time); base-policy episodes `episode_0/1/2.mp4` and final
checkpoint rollouts `condition00/10/20.mp4` from the evaluation (0.3-0.5 MB each, 8x time);
real-speed re-renders of three successful rollouts via `render_1x.py`; kinematic previews
`cond000_ep02_*`, `cond020_ep07_*` (0.5-1.5 MB).

Real-robot session 2026-08-14: thirteen compressed episode videos (960 px wide, 0.2-0.5 MB
each) and the full-size originals (22-31 MB each), subject to the lab's permission; the
demo video is already public on Google Drive (link above).
