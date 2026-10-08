# eval

Simulation evaluation of a fine-tuned checkpoint on held-out object placements. The
evaluation loop is PolaRiS's own `scripts/eval.py` with its `DroidJointPos` client, unchanged.

- `eval_policy.py` wraps PolaRiS's `AppLauncher` so that our environment is registered after
  Isaac starts and before the environment config is parsed, and raises the episode horizon to
  70 s (`POUR_EVAL_HORIZON_S`): 38% of the training episodes are longer than PolaRiS's 30 s
  default, which would truncate them before the pour and report a number about the horizon,
  not the policy. Use `eval_conditions.json` (held out), never the environment's
  `initial_conditions.json` (the training layouts).
- `shard_conditions.py` splits a conditions file into contiguous shards and merges the shard
  CSVs back with the original condition indices, which is the only way to run PolaRiS's
  single-file evaluation on several GPUs without touching it.
- `run_eval.sh <step> <server-gpu> <sim-gpu>...` serves one checkpoint and launches one
  simulator per GPU against it (30 placements: over two hours serially, 45 minutes on three
  GPUs, 15 on six). Shards start staggered because the server's first JIT compile blocks its
  event loop long enough to time out a simultaneous websocket handshake.
- `analyse_eval.py <eval_results.csv> <conditions.json>` joins each outcome with its layout
  and separates "never grasped" from "grasped but did not pour" by the rubric's progress.
- `log_action_rates.py` records every action the client hands the environment and reports
  the implied joint velocities against the FR3 ceiling (the stored videos are 8x fast and say
  nothing about speed); `--analyse <actions.npz>` re-reads a recording without a simulator.
- `render_1x.py` re-runs one episode rendering every step so the clip plays at real speed.
- `make_report.py` builds the evaluation report page from the result CSVs and clips.

Results of the checkpoint sweep are in `results/phase5_eval/`.
