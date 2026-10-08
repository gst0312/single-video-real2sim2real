"""Does our RLDS come out of openpi's loader in the same units the base model was trained in?

This is a hard gate and must be run on real data, not read off the code: an action-unit
or convention error here is silent - training converges on something, the policy just
moves wrong on the robot. An earlier lab pipeline lost a real-robot run to exactly that
class of bug (raw rad/s executed as normalised commands).

Three things are checked, in the order they would go wrong:

1. what the dataset hands over. Should be ABSOLUTE joint angles, inside the FR3-and-Panda
   intersection, with a binary gripper.
2. what `DeltaActions(make_bool_mask(7, -1))` leaves. openpi pushes this transform itself
   for the JOINT_POSITION action space: the seven arm dimensions become per-step deltas
   against the current state while the gripper stays absolute. Deltas at 15 Hz should be
   small - the motion budget caps them at 0.105-0.134 rad.
3. what normalisation with the OFFICIAL norm stats leaves. This is the real question: the
   stats come from the base checkpoint's own assets and are never recomputed, so if our
   actions were in different units the normalised values would come out far off unit
   scale. A healthy result is roughly zero-centred with a spread of order one.

    $POLARIS_ROOT/.venv/bin/python scripts/datagen/check_units.py \
        --config pi05_droid_jointpos_polaris_pourmustard
"""

import argparse

import numpy as np


def describe(name, x):
    x = np.asarray(x)
    flat = x.reshape(-1, x.shape[-1])
    print(f"\n{name}  shape {x.shape}")
    print("        " + "".join(f"{f'j{i+1}':>9}" for i in range(7)) + f"{'grip':>9}")
    for label, v in (("min", flat.min(0)), ("mean", flat.mean(0)), ("max", flat.max(0)),
                     ("std", flat.std(0))):
        print(f"  {label:<6}" + "".join(f"{val:9.3f}" for val in v[:8]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_droid_jointpos_polaris_pourmustard")
    p.add_argument("--batches", type=int, default=2)
    args = p.parse_args()

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(args.config)
    print(f"config {cfg.name}")
    print(f"  norm stats from {cfg.data.assets.assets_dir} (asset_id {cfg.data.assets.asset_id})")
    print(f"  rlds dir {cfg.data.rlds_data_dir}, action space {cfg.data.action_space}")

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = data_config.norm_stats
    if stats is None:
        raise SystemExit("no norm stats resolved - the assets path is wrong, stop here")
    a = stats["actions"]
    print(f"\nofficial norm stats for actions: mean {np.round(np.asarray(a.mean)[:8], 3)}")
    print(f"                                  std  {np.round(np.asarray(a.std)[:8], 3)}")

    loader = _data_loader.create_data_loader(cfg, num_batches=args.batches, shuffle=False)
    for obs, actions in loader:
        acts = np.asarray(actions)
        describe("actions as the model sees them (delta on 7 joints, then normalised)", acts)
        state = np.asarray(obs.state)
        describe("state (joint position + gripper), normalised", state[:, None, :])
        print("\nimages:", {k: tuple(np.asarray(v).shape) for k, v in obs.images.items()})
        print("image masks:", {k: bool(np.asarray(v).flat[0]) for k, v in obs.image_masks.items()})
        print("prompt sample:", obs.tokenized_prompt is not None)
        break

    print("\n--- what to look for ---")
    print("normalised actions roughly zero-centred, |values| mostly under ~3;")
    print("a spread far from that means our labels are not in the base model's units.")


if __name__ == "__main__":
    main()
