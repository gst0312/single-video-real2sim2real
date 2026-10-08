"""Turn physics-replayed episodes into a DROID-RLDS dataset openpi can train on.

plan §2's third block. The schema is the official PolaRiS cotrain dataset
(`owhan/PolaRiS-datasets`, read field by field in docs/data_format.md), and the builder
follows the official DROID RLDS builder template
(`droid_dataset_builder/droid/droid.py`): the same
`tfds.features` shapes, the same jpeg-encoded 180x320 images, the same episode_metadata
pair. Two places where that local template is thinner than what we need, both because
openpi's reader wants them (`training/droid_rlds_dataset.py:131-140`):

- `exterior_image_2_left` has to exist: the reader picks one of the two exterior images at
  random per trajectory. We only have one exterior camera, so both fields carry the same
  frame and the random pick becomes an identity - which is also what the policy sees at
  evaluation time, where the client sends one exterior view.
- all three `language_instruction*` fields have to exist: the reader samples one of them.
  The official cotrain data puts the same sentence in all three, and so do we.

What the reader actually consumes, and therefore what has to be right:
actions are `action_dict.joint_position` ⊕ `action_dict.gripper_position` - NOT the
`action` field - so those two are the labels. They are absolute joint angles: the training
side turns them into deltas itself (`DeltaActions(make_bool_mask(7, -1))`) and undoes it on
the way out, so anything pre-differenced here would be differenced twice.

`episode_metadata.file_path` and `recording_folderpath` must be unique per episode: the
reader builds `recording_folderpath + "--" + file_path + "--" + step` as a frame id for the
filter table. We ship no filter file (openpi then uses a table whose default is True, so
every frame passes), but the ids still have to be distinct.

    $RLDS_VENV/bin/python scripts/datagen/build_rlds.py \
        --episodes $WORK/traj/rollouts \
        --out $WORK/rlds
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

# the pure-numpy pieces live in the package so they can be tested without the heavy
# dependencies; make the repository's src importable when the package is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polaris_lfhv.physics_gate import CUP_MAX, GAP_MAX, LIFT_MIN, physics_gate_reasons  # noqa: E402

IMAGE_RES = (180, 320)          # height, width; the official cotrain dataset's size
NAME = "pour_mustard"


def episode_files(root, lift_min, gap_max, cup_max):
    """The observation npz that passed the physics gate.

    `--save-obs` writes an npz for every replayed episode, the failures included, so the
    gate has to be applied here or the dataset would teach the policy the misses too. The
    criteria are the ones the physics gate is for (2026-08-13 user's call): did it actually
    grasp, was the bottle knocked askew, did anything collide. Whether the pour would have
    landed in the cup is deliberately NOT a criterion - r2r2r's rigid-follow assumption does
    not model contact compliance or liquid, so it is not a fair thing to drop data over.

    Read from the report json the replay writes next to each npz; the criteria themselves
    are `polaris_lfhv.physics_gate`, shared with `analyse_dataset.py`.
    """
    kept, dropped = [], []
    for obs in sorted(Path(root).glob("*_obs.npz")):
        report = obs.with_name(obs.name[: -len("_obs.npz")] + ".json")
        if not report.exists():
            dropped.append((obs.name, "no report"))
            continue
        why = physics_gate_reasons(json.loads(report.read_text()), lift_min, gap_max, cup_max)
        (kept if not why else dropped).append((obs, "; ".join(why)) if why else obs)
    print(f"{len(kept)} episodes pass the physics gate, {len(dropped)} dropped")
    for name, why in dropped[:10]:
        print(f"  dropped {Path(name).name}: {why}")
    if len(dropped) > 10:
        print(f"  ... and {len(dropped) - 10} more")
    return kept


def build_steps(path):
    d = np.load(path, allow_pickle=True)
    ext, wrist = d["external_cam"], d["wrist_cam"]
    joints = d["joint_position"].astype(np.float64)
    grip_obs = np.asarray(d["gripper_position"], np.float64).reshape(-1, 1)
    action = d["action"].astype(np.float64)              # 7 joints + binary gripper
    lang = str(d["instruction"])
    n = len(action)
    assert ext.shape[1:3] == IMAGE_RES, f"{path}: images are {ext.shape[1:3]}, want {IMAGE_RES}"
    assert len(ext) == len(wrist) == len(joints) == len(grip_obs) == n, f"{path}: ragged"

    for i in range(n):
        yield {
            "observation": {
                "exterior_image_1_left": ext[i],
                # one exterior camera, so the reader's random pick between the two is an
                # identity and matches what the eval client sends
                "exterior_image_2_left": ext[i],
                "wrist_image_left": wrist[i],
                "joint_position": joints[i],
                "gripper_position": grip_obs[i],
            },
            "action_dict": {
                "joint_position": action[i, :7],
                "gripper_position": action[i, 7:8],
            },
            "action": action[i],
            "discount": np.float32(1.0),
            # the episodes in here have already passed the physics gate, so every one is a
            # demonstration and the reward lands on its last step, as in the DROID template
            "reward": np.float32(i == n - 1),
            "is_first": i == 0,
            "is_last": i == n - 1,
            "is_terminal": i == n - 1,
            "language_instruction": lang,
            "language_instruction_2": lang,
            "language_instruction_3": lang,
        }


class PourMustard(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Physics-replayed pour-mustard episodes from a single demo."}

    def __init__(self, *, episodes_dir, gate, **kwargs):
        self._episodes_dir = episodes_dir
        self._gate = gate
        super().__init__(**kwargs)

    @classmethod
    def get_metadata(cls):
        """TFDS normally reads description/citation from files next to the builder.

        That assumes the builder sits in its own package directory, which is how `tfds new`
        and the official DROID builder are laid out. This one is a single script, so the
        metadata is given here instead of on disk; without it TFDS tries to list the .py
        file as if it were a directory.
        """
        from tensorflow_datasets.core import dataset_metadata

        return dataset_metadata.DatasetMetadata(
            description="Pour-mustard episodes synthesised from one human demonstration and "
                        "replayed with physics in PolaRiS, in the DROID RLDS schema.",
            citation="", tags=[])

    def _info(self):
        img = lambda doc: tfds.features.Image(  # noqa: E731
            shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg", doc=doc)
        vec = lambda n, doc: tfds.features.Tensor(shape=(n,), dtype=np.float64, doc=doc)  # noqa: E731
        return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset({
                "observation": tfds.features.FeaturesDict({
                    "exterior_image_1_left": img("Exterior camera, left viewpoint"),
                    "exterior_image_2_left": img("Same camera; we have only one exterior view"),
                    "wrist_image_left": img("Wrist camera, left viewpoint"),
                    "joint_position": vec(7, "Measured joint position"),
                    "gripper_position": vec(1, "Measured gripper position, 0 open 1 closed"),
                }),
                "action_dict": tfds.features.FeaturesDict({
                    "joint_position": vec(7, "Commanded absolute joint position"),
                    "gripper_position": vec(1, "Commanded gripper position, binary"),
                }),
                "action": vec(8, "7x absolute joint position, 1x gripper"),
                "discount": tfds.features.Scalar(dtype=np.float32),
                "reward": tfds.features.Scalar(dtype=np.float32),
                "is_first": tfds.features.Scalar(dtype=np.bool_),
                "is_last": tfds.features.Scalar(dtype=np.bool_),
                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                "language_instruction": tfds.features.Text(),
                "language_instruction_2": tfds.features.Text(),
                "language_instruction_3": tfds.features.Text(),
            }),
            "episode_metadata": tfds.features.FeaturesDict({
                "file_path": tfds.features.Text(doc="Unique per episode; the reader keys on it"),
                "recording_folderpath": tfds.features.Text(doc="Unique per episode, likewise"),
            }),
        }))

    def _split_generators(self, dl_manager):
        paths = episode_files(self._episodes_dir, *self._gate)
        if not paths:
            raise SystemExit(f"no *_obs.npz under {self._episodes_dir}")
        print(f"{len(paths)} episodes")
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths):
        for path in paths:
            key = path.name[: -len("_obs.npz")]
            yield key, {
                "steps": list(build_steps(path)),
                # distinct per episode, which is all the reader needs from these
                "episode_metadata": {"file_path": f"success_{NAME}/{key}.npz",
                                     "recording_folderpath": f"success_{NAME}/{key}"},
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True, help="directory of *_obs.npz")
    p.add_argument("--out", required=True, help="tfds data_dir to write into")
    p.add_argument("--lift-min", type=float, default=LIFT_MIN,
                   help="the bottle has to come up this far, i.e. it was grasped; "
                        "same threshold as the rubric's lift criterion")
    p.add_argument("--gap-max", type=float, default=GAP_MAX,
                   help="how far the bottle may drift from the reference before it "
                        "counts as knocked askew rather than carried")
    p.add_argument("--cup-max", type=float, default=CUP_MAX,
                   help="cup displacement that counts as a collision")
    args = p.parse_args()

    builder = PourMustard(episodes_dir=args.episodes, data_dir=args.out,
                          gate=(args.lift_min, args.gap_max, args.cup_max))
    builder.download_and_prepare()
    print(f"wrote {args.out}/{NAME}/1.0.0")
    print(builder.info)


if __name__ == "__main__":
    main()
