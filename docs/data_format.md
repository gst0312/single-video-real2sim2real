# Training data format: what the official co-training dataset looks like

The RLDS writer (`scripts/datagen/build_rlds.py`) follows the dataset openpi's PolaRiS
configuration was trained on, read field by field rather than from `features.json` alone.

The reference is `polaris_droid_cotrain_dataset/1.0.0` in HuggingFace `owhan/PolaRiS-datasets`:
TFDS tfrecord format, 32 shards, 316 episodes, 2.9 GB. The numbers below were obtained by
reading it with `tensorflow-datasets`.

## Fields

An episode has `episode_metadata` and `steps`.

`episode_metadata` holds two strings, `file_path` and `recording_folderpath`. In the official
data they are equal and look like
`success_/media/.../projects/real2simeval/demos/kanav_cloth_chairs/episode_14.npz`. They are
not for humans: openpi's reader builds `recording_folderpath + "--" + file_path + "--" + step`
as a per-frame id for the filter table, so the pair must be unique per episode.

Each step carries:

- `action`, float64, 8-d, documented as "7x jp, 1x gripper pos".
- `action_dict.joint_position` (float64, 7, "Commanded joint position") and
  `action_dict.gripper_position` (float64, 1, "Commanded gripper position"). Measured:
  `action` is exactly their concatenation.
- `observation.joint_position` (float64, 7), `observation.gripper_position` (float64, 1).
- `observation.exterior_image_1_left`, `observation.exterior_image_2_left`,
  `observation.wrist_image_left`: 180×320×3 uint8, JPEG encoded.
- `language_instruction`, `language_instruction_2`, `language_instruction_3`: three text
  fields with identical content in the official data.
- `is_first`, `is_last`, `is_terminal` (bool), `reward`, `discount` (float32).

Instructions are lower-case natural phrases without punctuation (`put the marker in the mug`,
`put the lid on the pan`, `stack the blocks`, `turn the block 90 degrees`). Ours,
`pour the mustard into the blue cup`, is in the same style.

## Measured characteristics

Over 40 sampled episodes: 46 to 593 steps, mean 187 (3 to 40 s at 15 Hz).

Observed joint angles always lie inside the Panda's limits (e.g. j6 max 3.7526 against a
limit of 3.7525; j3 min -2.8973 exactly at the limit). Commanded actions are not bounded:
j1 reaches 4.6457, j3 -4.3560, j4 1.3724, all outside the Panda's position limits. The
official data records the raw command and lets the simulator clamp it. Our labels are
reference trajectories that have already been gated to the FR3-and-Panda intersection, so our
data is cleaner on this axis; that is a distribution difference worth knowing, not a conflict.

The gripper in the official data is continuous, not binary: observed `gripper_position`
spans -0.00086 to 1.00002 (noisy, slightly out of range) and the commanded value takes
continuous values in [0, 1] (0.005, 0.024, 0.028, ...). PolaRiS's `droid_jointpos_client.py`
binarises the policy output at 0.5 before it reaches the environment, so "continuous policy
output, binary execution" is the normal state of that chain.

## What openpi's reader consumes

`src/openpi/training/droid_rlds_dataset.py:restructure` takes the action from
`action_dict["joint_position"]` concatenated with `action_dict["gripper_position"]` (not from
`action`); picks one of `exterior_image_1_left` / `exterior_image_2_left` at random per
trajectory; always uses `wrist_image_left`; and samples one of the three instruction fields.

With `action_space = JOINT_POSITION`, `RLDSDroidDataConfig` pushes
`DeltaActions(make_bool_mask(7, -1))` into the data transforms: the seven absolute joint
positions become deltas against the current joint state for training and `AbsoluteActions`
restores them on output; the eighth dimension stays absolute. Writing absolute joint
positions to disk is therefore correct, and pre-differencing would be differenced twice.

`filter_dict_path` names a JSON mapping each episode to the frame range to keep; both official
datasets ship one.

## Decisions taken for our writer

Field names, dtypes, image size (180×320×3 JPEG), `action` consistent with `action_dict`,
unique `episode_metadata` strings, lower-case instruction: all as above. The three open
items were settled as follows (2026-08-13):

- Gripper label: the binary 0/1 command we actually sent (PolaRiS's client binarises before
  the environment, and the sim executes a binary gripper).
- No filter file: openpi then builds a lookup table whose default is True, so every frame
  passes (`droid_rlds_dataset.py:110-113`).
- One exterior camera: the same frame is written to both exterior fields, so the reader's
  random choice is an identity, which matches evaluation (the client sends one exterior
  view, and `droid_policy.py` reads only `exterior_image_1_left`).

`scripts/datagen/check_units.py` verifies the result end to end through openpi's loader: the
absolute angles out of the dataset, the deltas after `DeltaActions`, and the normalised
values under the official norm stats, which must come out roughly zero-centred with a spread
of order one. On the generated dataset: after DeltaActions and normalisation the action
means were within -0.068..0.064, the standard deviations 0.152..0.292, and all values within
±2.1.
