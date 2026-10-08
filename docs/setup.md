# Setup and reproduction notes

The pipeline runs across several Python environments because its upstreams do not share
one: PolaRiS (IsaacLab 2.3.0, Python 3.11) and Real2Render2Real's synthesis stack (jax
0.5.3, Python 3.12) cannot live in the same venv. This page records what was installed, the
two deviations from PolaRiS's `pyproject.toml` that were unavoidable, and the environment
variables the scripts read instead of hard-coded paths.

Everything below was run on one machine: Ubuntu 24.04.4, NVIDIA driver 575.57.08 (CUDA
12.9), nvcc 12.9.86, eight RTX 6000 Ada (48 GB each), uv 0.9.26. Training used four of the
GPUs, simulation evaluation three to six.

## Environment variables

Scripts and shell wrappers take locations from the environment; nothing in the repository
assumes a particular machine.

| Variable | Meaning |
|---|---|
| `POLARIS_ROOT` | PolaRiS checkout (github.com/arhanjain/polaris, commit 129abc4) with its `.venv` and `PolaRiS-Hub/` |
| `POLARIS_3DGS` | the `gsplat-3dgs-renderer` branch checkout (zubair-irshad/polaris, baseline 1e3a6d4) with this project's SH / background / principal-point patches |
| `WORK` | working directory for generated artifacts: synthesised episodes, replays, RLDS, evaluation runs |
| `GSWORLD_ROOT` | GSWorld checkout; source of the scene splat, link labels, object meshes, camera calibration and real recordings (not redistributed, see `docs/gsworld_to_polaris.md`) |
| `TWODGS` | hbb1/2d-gaussian-splatting checkout (335ad61), used by the splat conversion and the render-error tools for its ply readers and PSNR/SSIM |
| `TRAJ_VENV` | the offline synthesis venv (jax, jaxmp, jaxls, trajgen) |
| `RLDS_VENV` | the venv that reads and writes RLDS (tensorflow-cpu, tensorflow-datasets) |
| `DEMO_TRACK` | the tracked object trajectory of the human demonstration (`take4.npy`, 2 objects × T × (xyz + quaternion)); not included |
| `RLDS_DATA_DIR` | where `build_rlds.py` wrote; read by the training config (defaults to `$WORK/rlds`) |
| `DROID_ROOT`, `NUC_USER`, `NUC_CONTAINER`, `NUC_SUDO_PASSWORD` | real-robot launcher only, see `docs/real_robot.md` |

## PolaRiS stack

Install as PolaRiS's README says, into its own checkout (the venv is 19 GB and the README's
commands assume you stand in that repository). Its submodules use SSH URLs; if plain
`github.com` SSH is not set up, rewrite them:

```bash
git clone --branch main https://github.com/arhanjain/polaris "$POLARIS_ROOT"
cd "$POLARIS_ROOT"
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
```

Submodules at the time: glm (5c46b9c) and openpi (bd70b8f, Physical-Intelligence/openpi).

Two deviations from the official `pyproject.toml` were required on this machine:

1. torch index `cu130` → `cu129`. The README says to lower torch and its index for older CUDA;
   the 575 driver supports CUDA 12.9, the cu130 wheels do not run, and the splat rasteriser
   is JIT-compiled on first import with the local nvcc, which fails on a major-version
   mismatch. Installed: torch 2.13.0+cu129.
2. Pin an old setuptools for one package:

   ```toml
   [tool.uv.extra-build-dependencies]
   flatdict = ["setuptools<81"]
   ```

   isaaclab 2.3.0 depends on flatdict 4.0.1, which ships as sdist only and imports
   `pkg_resources` in its `setup.py` without declaring it; setuptools ≥ 81 no longer
   provides `pkg_resources`. The pin affects only flatdict's build.

Then:

```bash
export UV_CACHE_DIR=<same disk as the venv>     # uv can hard-link instead of copying
cd "$POLARIS_ROOT" && uv sync
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub   # 1.7 GB
```

Two run-time requirements: `OMNI_KIT_ACCEPT_EULA=YES` (Isaac Sim's first start otherwise
asks interactively) and run through `uv run`, not `.venv/bin/python` directly (the
rasteriser's JIT build needs `ninja` on `PATH`, which only `uv run` provides).
`scripts/polaris_env.sh` sets both, selects the emptiest GPU unless `CUDA_VISIBLE_DEVICES`
is set, and switches to the gsplat renderer; every Isaac-side script in this repository is
run through it.

## openpi side

Installed inside PolaRiS's submodule as its README says:

```bash
cd "$POLARIS_ROOT/third_party/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Serving the base policy:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_droid_jointpos_polaris \
  --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
```

The weights (11.6 GB) download to `~/.cache/openpi`; the norm stats load from the
checkpoint's assets directory. They are the official statistics and are never recomputed.

Two small changes to openpi were needed and are documented in `docs/provenance.md`:
`transforms.DeltaActions` copies before subtracting (RLDS batches are read-only numpy arrays),
and the registration of this project's training config in `training/config.py` is wrapped in
`try/except ImportError` so openpi entry points still start when this repository is not on
`PYTHONPATH`. Registering the config is two lines next to the existing
`*polaris_config.get_polaris_configs(),`:

```python
import polaris_lfhv.training.pour_mustard_config as pour_mustard_config
...
*pour_mustard_config.get_configs(),
```

## Offline synthesis venv (`TRAJ_VENV`)

Python 3.12 (jaxmp's jaxls dependency needs it). Pinned as Real2Render2Real's stack was
verified: jax 0.5.3, numpy 1.26.4; jaxls (chungmin99/jaxls, 21219e08: the pre-rename
interface jaxmp expects), jaxmp (uynitsuj/jaxmp, d1ab9c7: rsrd's submodule source), trajgen
(uynitsuj/trajgen, 6b19f56: r2r2r's submodule source), all installed from local checkouts.
Known traps: jaxmp pins mujoco-mjx 3.2.3, which needs mujoco 3.2.3 (uv resolves to 3.11 and
the import fails); jax 0.5.3 needs cuDNN 9.8 (9.24 fails on driver 575 with
`CUDNN_STATUS_INTERNAL_ERROR`); torch should be the CPU build here (only trajgen's small
arithmetic uses it, and the default cu13 build drags CUDA 13 wheels into a cu12 venv). The
FR3 URDF comes from `robot_descriptions` (`fr3_description`, upstream
frankarobotics/franka_description); franka_description 2.8.1 ships xacro only.

## RLDS venv (`RLDS_VENV`)

Package versions follow openpi's `rlds` dependency group: tensorflow-cpu 2.15.0,
tensorflow-datasets 4.9.9, plus protobuf 4.25.8 and tensorflow-metadata 1.17.1 as in openpi's
lock file (the default resolution fails to import on a protobuf interface change). openpi's
loader additionally needs `dlimp`.

## Things that mislead

- With `use_fabric=True` the USD stage does not follow joint motion: IsaacLab writes link
  poses to Fabric and never back to USD, so `UsdGeom.XformCache` returns the default pose
  whatever joint state was written. Measured: two very different qpos, same panda_link7 mesh
  origin from USD (0.3597, 0, 0.5969), while the articulation data moved correctly from
  (0.360, 0, 0.597) to (-0.068, 0.455, 0.732). A fit based on USD points has a constant cost
  and "converges" to its initial value. Link poses must come from `robot.data.body_pos_w` /
  `body_quat_w`; USD is only used for the static mesh-to-body transform. Scripts that pose the
  robot carry a self-check: change the pose, the model points must move.
- Isaac Sim swallows Python's stdout once running. Results are written to files (json/npy),
  never read off the console.
- Isaac Sim's exit code is not trustworthy: a script can raise, print its traceback and still
  return 0. Success is judged by the artifact (`eval_results.csv` exists), never by `$?`.
- Long jobs must run in their own session (`setsid nohup ... &`); a job started from a
  foreground task group was killed by SIGTERM when the group was reclaimed.
- GPU selection: `nvidia-smi` for a free card, and `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the
  numbering matches what `nvidia-smi` shows.
- Starting several simulator clients against one freshly started policy server: the first
  inference triggers an XLA JIT compile that blocks the server's event loop long enough for a
  later client's websocket handshake to time out. `scripts/eval/run_eval.sh` waits for each
  shard to take its first step before launching the next.

## What is not pip-installable from this repository

`pyproject.toml` installs the pure-Python package (numpy, scipy) that the tests and the
example use, plus optional `synthesis` and `rlds` groups for the two offline venvs. PolaRiS,
IsaacLab, Isaac Sim, the gsplat renderer branch, openpi and GSWorld are external checkouts
located through the variables above.
