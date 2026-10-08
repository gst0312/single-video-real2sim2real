"""Our training config: PolaRiS's official one, with our dataset and LoRA.

This is a copy of `pi05_droid_jointpos_polaris` from openpi's
`src/openpi/training/misc/polaris_config.py` (verified byte-identical to
Physical-Intelligence/openpi main at 15a9616 on 2026-08-13), changed in as few places as
possible. Everything not listed below - the learning rate schedule, warmup, action horizon,
`num_workers=0`, the JOINT_POSITION action space and the DeltaActions it implies - is
theirs untouched.

What changes, and why each one:

1. `datasets` points at our RLDS instead of DROID + the cotrain mix. First round is the
   minimal closed loop; DROID's RLDS is terabytes and co-training is a separate decision.
   No `filter_dict_path`: openpi then builds a lookup table whose default is True, so every
   frame passes (`droid_rlds_dataset.py:110-113`).
2. `rlds_data_dir` is where `scripts/datagen/build_rlds.py` wrote.
3. `num_train_steps` 1000 -> 10000 (user's call).
4. LoRA. The official config is a FULL fine-tune, and switching to LoRA is three changes
   that must agree with each other, none of which fails loudly if you forget it:
   the two `*_variant="..._lora"` fields on the model; a `freeze_filter` built from a
   Pi0Config carrying THE SAME arguments (the default is `nnx.Nothing`, i.e. nothing is
   frozen and you have silently done a full fine-tune); and `ema_decay=None`, which
   openpi's own LoRA example spells out. LFHV's last round used exactly this triple, which
   is the cross-check that it is right.
5. `batch_size` 128 -> 32. The official 128 is not a free-standing choice: it comes with
   that config's `datasets`, which co-train on DROID (weight 0.9) plus the cotrain mix -
   a dataset of a completely different order from our ~170 episodes. Every reference that
   fine-tunes a single task on a small set uses 32: openpi's own `TrainConfig` default is
   32, all of LFHV's pour and place configs are 32, and r2r2r's reported π0-FAST recipe is
   32 as well (second-hand - their repo points policy training at an external openpi fork,
   so that one could not be verified here). 32 with the official lr 5e-5 is exactly what
   LFHV ran, so the pairing has a precedent.

   `batch_size` is the GLOBAL batch and `scripts/train.py:198` asserts it divides the
   device count; with `fsdp_devices=1` the mesh is `(devices, 1)`, i.e. plain data parallel
   with the model replicated (`training/sharding.py:22`). At 32 there is little point
   spreading over all eight cards - four samples each leaves the GPUs mostly idle between
   all-reduces - so the run uses four, eight samples each, which also leaves cards free for
   the sim evaluation of intermediate checkpoints.

`action_horizon` stays 15, NOT the 16 LFHV used: it has to match the checkpoint being
loaded, and ours is PolaRiS's joint-position DROID model while theirs was pi05_droid.
The norm stats path follows the same rule - the base checkpoint's own assets, never
recomputed.

The wrist camera is in, because pouring ends with aiming the spout into the cup and the
third-person view is occluded by the arm at exactly that moment. Nothing to configure:
`RLDSDroidDataConfig` feeds `wrist_image_left` by default.

Register it by adding these two lines to openpi's `training/config.py`, next to the
existing `*polaris_config.get_polaris_configs(),`:

    import polaris_lfhv.training.pour_mustard_config as pour_mustard_config
    ...
    *pour_mustard_config.get_configs(),
"""

import os

RLDS_DATA_DIR = os.environ.get("RLDS_DATA_DIR", os.path.join(os.environ.get("WORK", "work"), "rlds"))
BASE = "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"


def get_configs():
    import openpi.models.pi0_config as pi0_config
    import openpi.training.droid_rlds_dataset as droid_rlds_dataset
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import AssetsConfig
    from openpi.training.config import RLDSDroidDataConfig
    from openpi.training.config import TrainConfig

    lora = dict(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    model_args = dict(action_horizon=15, pi05=True, **lora)

    return [
        TrainConfig(
            name="pi05_droid_jointpos_polaris_pourmustard",
            model=pi0_config.Pi0Config(**model_args),
            data=RLDSDroidDataConfig(
                # inert on the RLDS path - the data comes from rlds_data_dir and the
                # asset id is set explicitly - but tyro makes it required, so the
                # official configs have to pass --data.repo-id on the command line.
                # Setting it here keeps the launch command to just the config name.
                repo_id="pour_mustard",
                assets=AssetsConfig(assets_dir=f"{BASE}/assets", asset_id="droid"),
                datasets=(
                    droid_rlds_dataset.RLDSDataset(
                        name="pour_mustard", version="1.0.0", weight=1.0),
                ),
                rlds_data_dir=RLDS_DATA_DIR,
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(f"{BASE}/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
            num_train_steps=10_000,
            batch_size=32,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # the RLDS loader does its own multiprocessing
            # LoRA: the model variants, a matching freeze filter, and no EMA
            freeze_filter=pi0_config.Pi0Config(**model_args).get_freeze_filter(),
            ema_decay=None,
        ),
    ]
