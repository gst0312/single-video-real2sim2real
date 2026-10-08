# real_robot

Running the simulation-trained checkpoint on the Franka FR3 through the DROID client, with
the position-control conventions the checkpoint was trained for. Procedure, alignment checks
and the list of client patches are in `docs/real_robot.md`; the session log is in
`results/real_robot_20260814/`.

- `run_eval_safe.py` launches DROID's `evaluate_openpi.py` with its source patched in memory
  (joint-position action space, no ±1 clip, 1050-step horizon, binary gripper, hold while the
  gripper closes, lock the gripper on a confirmed grasp, scoring fix), works around the NUC
  gripper start-up race, and runs an abort-only motion watchdog. Credentials and machine
  names come from `DROID_ROOT`, `NUC_USER`, `NUC_CONTAINER`, `NUC_SUDO_PASSWORD`.
  `--show-diff` prints what would run; `--check-only` brings the stack up without an episode.
- `check_placement.py <first frame or dual-view video>` reads the bottle's apparent size in
  the wrist view and says whether the placement is inside the training distribution
  (calibrated on the 30 held-out sim episodes and all 13 real episodes of the first session).
- `compare_obs.py` asks the served policy for an action chunk from the home pose twice, once
  with sim images and once with real images, and reports how much the first commanded step
  differs; it needs `openpi_client` (`OPENPI_CLIENT_SITE`).
- `archive_rollouts.py` re-encodes new real-robot recordings (`ROLLOUTS`, `EVAL_CSVS`) into a
  browsable session directory with first frames and an `episodes.csv` index; idempotent.
