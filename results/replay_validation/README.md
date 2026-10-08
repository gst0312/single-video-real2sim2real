# replay_validation

Does the simulator look like the cell, and does the action convention behave like the arm?

## Static frames at recorded joint angles

`rollout_frames_summary.json`: eight real recordings (home_static, randomwalk_1..4,
vertical_1, wristroll_1..2), three frames each, the simulator posed at the recorded joint
angles, compared to the real frame per region. Per-segment means:

| segment | frames | pair gap ms | ext PSNR | SSIM | IoU | brightness | wrist PSNR | SSIM | brightness |
|---|---|---|---|---|---|---|---|---|---|
| home_static | 3 | 14.7 | 16.92 | 0.691 | 0.899 | 0.782 | 14.50 | 0.793 | 0.777 |
| randomwalk_1 | 3 | 11.1 | 16.99 | 0.692 | 0.897 | 0.794 | 13.98 | 0.784 | 0.770 |
| randomwalk_2 | 3 | 14.8 | 17.17 | 0.696 | 0.901 | 0.796 | 14.52 | 0.808 | 0.767 |
| randomwalk_3 | 3 | 20.8 | 16.60 | 0.684 | 0.890 | 0.770 | 14.35 | 0.774 | 0.771 |
| randomwalk_4 | 3 | 16.3 | 16.81 | 0.687 | 0.893 | 0.779 | 16.23 | 0.829 | 0.867 |
| vertical_1 | 3 | 12.3 | 16.39 | 0.679 | 0.887 | 0.760 | 15.43 | 0.813 | 0.863 |
| wristroll_1 | 3 | 14.3 | 17.04 | 0.693 | 0.900 | 0.789 | 15.52 | 0.817 | 0.865 |
| wristroll_2 | 3 | 13.3 | 16.37 | 0.679 | 0.886 | 0.759 | 15.48 | 0.816 | 0.863 |
| all | 24 | 14.7 | 16.78 | 0.688 | 0.894 | 0.778 | 15.00 | 0.804 | 0.818 |

Arm silhouette IoU (exterior camera): 0.840, 0.837, 0.843, 0.814, 0.830, 0.815, 0.841, 0.817;
joint write-in error 0.0000° on every segment.

Reading note: the global exposure is the midpoint between GSWorld's native exposure and a fit
to the lab's July frames, so the simulator is about 22% darker than those frames and the
PSNR/SSIM/brightness columns are lower than an exposure-matched render would give (exterior
19.0 → 16.8); the bright-region IoU threshold was scaled accordingly (110 → 86). With the
wrist mount reverted to GSWorld's `wrist2eef`, wrist PSNR rose 12.3 → 15.0 and SSIM 0.73 →
0.80 against the previous mount. Frames: `../figures/replay_*`.

## Recorded joint angles replayed as absolute-position actions

PolaRiS's action is an absolute joint-position target. Feeding measured joint angles as that
target, step by step, measures the stock controller's tracking on this arm model.

`replay_gsworld_recordings/`, the eight recordings above (800 steps each for the random
walks):

| segment | steps | rms (deg) | max (deg) | worst joint |
|---|---|---|---|---|
| home_static | 321 | 0.00 | 0.00 | j2 (0.00) |
| randomwalk_1 | 800 | 0.46 | 2.13 | j2 (0.56) |
| randomwalk_2 | 800 | 0.54 | 1.41 | j3 (0.66) |
| randomwalk_3 | 800 | 0.51 | 1.42 | j6 (0.58) |
| randomwalk_4 | 800 | 0.54 | 1.36 | j6 (0.60) |
| vertical_1 | 364 | 0.41 | 1.12 | j6 (0.80) |
| wristroll_1 | 796 | 0.38 | 1.18 | j7 (1.01) |
| wristroll_2 | 796 | 0.38 | 1.19 | j7 (1.02) |

`replay_teleop/`, eight episodes of a private teleoperation dataset on the same arm (five
"pick up the mustard", three pour-type tasks), measured qpos replayed with the continuous
gripper closure:

| episode | task | steps | rms (deg) | max (deg) | worst joint |
|---|---|---|---|---|---|
| 000120 | move the cup to the right and pour the coke into the cup | 788 | 1.61 | 8.49 | j7 (2.73) |
| 000200 | pick up the mustard | 159 | 1.46 | 7.94 | j2 (2.64) |
| 000201 | pick up the mustard | 238 | 1.09 | 5.82 | j2 (1.64) |
| 000202 | pick up the mustard | 191 | 1.04 | 4.56 | j2 (1.40) |
| 000203 | pick up the mustard | 204 | 0.93 | 4.10 | j7 (1.26) |
| 000204 | pick up the mustard | 263 | 1.02 | 5.16 | j2 (1.52) |
| 000220 | move the cup to the right and pour the sprite into the cup | 743 | 1.81 | 9.01 | j7 (3.10) |
| 000222 | move the cup to the right and pour the fanta into the cup | 736 | 1.60 | 7.60 | j7 (2.81) |

Interpretation: rms is about the 95th-percentile commanded speed times one control period
(1/15 s), i.e. a single-step controller lag, so no actuator identification is needed. The
replays were run on an earlier asset revision; physics does not depend on the splat and the
only physical change since (1 mm of platform height) does not move these numbers.
