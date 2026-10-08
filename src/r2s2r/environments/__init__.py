"""Register our pour-mustard environment, following polaris/src/polaris/environments/__init__.py.

Importing this module registers the environment with gymnasium; it also pulls in
polaris.environments so the six official environments stay available in the same process.
Everything about the environment itself (robot, action space, cameras, 15 Hz rate) comes
from polaris's DroidCfg untouched. What is ours is the scene USD, the wrist camera, the
arm's joint limits (narrowed to the FR3-and-Panda intersection, see FR3_PANDA_LIMITS) and
the pour rubric in `rubrics.py`.
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.stage import get_current_stage
from pxr import Semantics

import polaris.environments  # noqa: F401  (registers the six official environments)
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import DATA_PATH

from r2s2r.environments.rubrics import pour_mustard_rubric


@configclass
class PourMustardCfg(DroidCfg):
    """PolaRiS's DROID config with the wrist camera this lab actually has.

    Everything else is theirs. The wrist camera is not, because it is a different camera on
    a different bracket: PolaRiS models DROID's, mounted at (0.011, -0.031, -0.074) off the
    gripper base with an 87.6 degree horizontal field of view, while the ZED Mini here
    reports fx 730.59 across 1280 px, which is 82.4 degrees. Apertures follow from keeping
    the 2.8 mm focal length PolaRiS uses.

    The mount is GSWorld's `wrist2eef`, and the near field is why: the Robotiq is rigid
    on link8 with model-true geometry, and only this mount renders it at the real
    apparent size (camera 8.4 cm from the gripper base). A 2026-08-13 attempt to
    "refine" the mount against `markers_in_base.json` replaced it with the 2026-07-07
    hand-eye (camera at 5.3 cm, gripper rendered ~1.6-1.9x too large) - that fit was
    circular: the marker file was itself solved through the wrist camera WITH the July
    hand-eye, so its "18.3 px error" for wrist2eef measured the bias of the reference,
    not the mount. Any future refinement must use truth independent of the wrist chain
    (ext-camera-solved markers, wrist depth vs the table plane) and must keep the
    rendered gripper scale as an acceptance gate. The July hand-eye, for reference:
    pos (0.014030, -0.014222, -0.053317) rot (-0.425071, 0.580047, 0.566073, -0.403015).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_cam.offset = CameraCfg.OffsetCfg(
            pos=(0.000744, -0.032081, -0.077805),
            rot=(-0.402458, 0.587241, 0.565950, -0.415784),
            convention="opengl",
        )
        self.scene.wrist_cam.spawn.horizontal_aperture = 4.905622
        self.scene.wrist_cam.spawn.vertical_aperture = 2.759413
        # The ZED Mini's principal point is a long way off centre (cx 598.55 of 1280), so a
        # camera that assumes it is centred draws the whole wrist view about 41 px sideways.
        # USD says it with an aperture offset: one pixel is focal/fx wide, and USD's image y
        # points up while the intrinsics' points down. GSWorld hit and fixed the same thing
        # (`gs_world_wrapper.py: cam_maniskill2gs`).
        self.scene.wrist_cam.spawn.horizontal_aperture_offset = (640 - 598.5459) * 2.8 / 730.5903
        self.scene.wrist_cam.spawn.vertical_aperture_offset = (357.7329 - 360) * 2.8 / 730.5903


class PourMustardEnv(ManagerBasedRLSplatEnv):
    """PolaRiS's environment, unchanged, with an option to blank the wrist view.

    The wrist camera now renders from our own mount and field of view, see PourMustardCfg.
    Blanking is kept only as a fallback, to be decided once we can see how the wrist view
    actually looks.

    Note for when that decision comes: openpi already has a first-class way to say a
    camera is absent. `policies/droid_policy.py` packs PI0/PI05 inputs with an
    `image_mask` per camera and ships `right_wrist_0_rgb` as zeros with the mask False.
    Blanking the pixels while leaving the mask True instead tells the model the wrist sees
    black, which is not the same thing.
    """

    WRIST_CAMERAS = ("wrist_cam",)
    #: Arm joint position limits, radians, in panda_joint1..7 order.
    #:
    #: PolaRiS ships NVIDIA's DROID cell, whose USD carries the Panda's limits; this lab
    #: runs an FR3. The two kinematics are identical (franka_description's fr3 and fer
    #: kinematics.yaml are byte for byte the same) but the limits are not, so data has to
    #: live in the intersection to be both executable on the FR3 and replayable here.
    #: FR3 is tighter in three places (j4 upper, j5 both, j6 lower by 0.46 rad) and wider
    #: in the other four. Numbers and their sources in docs/fr3_limits.md; the generator
    #: gates on the same array (scripts/datagen/synthesize_trajectories.py LIMITS_LOW/HIGH).
    #: Applied at runtime so PolaRiS's own noninstanceable.usd stays untouched.
    FR3_PANDA_LIMITS = np.array([
        [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.1169],
        [-2.8763, 2.8763], [0.4398, 3.7525], [-2.8973, 2.8973]])
    #: Optional per-channel gain and offset applied to the wrist image after compositing.
    #:
    #: Re-enabled 2026-08-13: the wrist composite renders ~14% brighter
    #: than its target (per-channel real/sim 0.876/0.874/0.885 measured on the July
    #: frames). The global relight now sits halfway between GSWorld's native exposure and
    #: the July fit (a split-the-difference choice against the darker February teleoperation
    #: footage), so the wrist keeps just the pure July correction on top.
    #: R2S2R_WRIST_RESPONSE=0 always renders the raw composite.
    WRIST_RESPONSE = (np.array([0.876, 0.874, 0.885], dtype=np.float32), 0.0)
    #: subtree drawn by the raytracer even when the rest of the robot comes from splats.
    #: The wrist camera sits 8 cm from the gripper while the splat was reconstructed from a
    #: metre away, so the gripper it draws is a smear exactly where the policy needs detail;
    #: the raytraced Robotiq lands within a few pixels of the real one and stays crisp.
    RAYTRACED_SUBTREES = ("Gripper",)

    def __init__(self, cfg, *args, blank_wrist: bool = False, robot_splat: bool = True,
                 raytraced_gripper: bool = True, wrist_response: bool = True,
                 fr3_limits: bool = True, usd_file: str | None = None, **kwargs):
        self.blank_wrist = blank_wrist
        # R2S2R_WRIST_RESPONSE=0 renders the raw composite, for measuring the gap again
        env_resp = os.environ.get("R2S2R_WRIST_RESPONSE")
        on = wrist_response if env_resp is None else env_resp not in ("0", "false", "False")
        self.wrist_response = self.WRIST_RESPONSE if on else None
        self.robot_splat = robot_splat
        # R2S2R_RAYTRACED_GRIPPER=0 puts the gripper back on the splats, which is what
        # GSWorld's own renders do; used when comparing against those renders directly.
        env_gripper = os.environ.get("R2S2R_RAYTRACED_GRIPPER")
        self.raytraced_gripper = (raytraced_gripper if env_gripper is None
                                  else env_gripper not in ("0", "false", "False"))
        if usd_file is not None:
            # PolaRiS always builds the scene with the robot drawn from its own per-link
            # splats. Those were scanned from NVIDIA's DROID cell and come out dark and
            # brown; measured against a real frame the arm they draw covers a fifth of the
            # bright pixels the real FR3 does, which is what reads as a ghost in a blend.
            # robot_splat=False is PolaRiS's own switch for drawing the robot with the
            # raytracer from its USD instead.
            self.usd_file = usd_file
            cfg.dynamic_setup(usd_file, robot_splat)
            usd_file = None
        super().__init__(cfg, *args, usd_file=usd_file, **kwargs)
        if self.robot_splat and self.raytraced_gripper:
            self._tag_raytraced_subtrees()
        if fr3_limits:
            self._apply_fr3_joint_limits()

    def setup_splat_robot(self):
        """PolaRiS's own routine, with the raytraced subtrees left out.

        Copied from `manager_based_rl_splat_environment.setup_splat_robot`; the only change
        is the filter. Skipping the load matters rather than just skipping the transform: a
        link splat is stored in its own frame and only moves when `transform_many` is given
        that prim's pose, so a loaded-but-untracked gripper would sit at the world origin.
        """
        if not self.robot_splat:
            return
        if not self.raytraced_gripper:
            super().setup_splat_robot()
            return
        more_splats = {}
        robot_asset_path = Path(self.cfg.scene.robot.spawn.usd_path).parent
        skipped = 0
        for ply in sorted(robot_asset_path.glob("SEGMENTED/*.ply")):
            if ply.stem.startswith(self.RAYTRACED_SUBTREES):
                skipped += 1
                continue
            more_splats[ply.stem] = ply
            self.views[ply.stem] = GeometryPrim(
                prim_paths_expr=f"/World/envs/env_0/robot/{ply.stem.replace('-', '/')}",
                reset_xform_properties=False,
            )
        print(f"[r2s2r] {len(more_splats)} link splats loaded, {skipped} left to the raytracer")
        self.splat_renderer.add_splats(more_splats)

    def _apply_fr3_joint_limits(self):
        """Narrow the arm's position limits to the FR3-and-Panda intersection at runtime.

        IsaacLab 2.3.0's `write_joint_position_limit_to_sim` writes `data.joint_pos_limits`,
        recomputes the soft limits and pushes the DOF limits to PhysX, so the shipped USD
        needs no edit and the stock behaviour is one flag away (`fr3_limits=False`). Without
        this the environment is more permissive than the real arm, and a replay that PolaRiS
        accepts could be one the FR3 refuses - j6 alone differs by 0.46 rad.
        """
        robot = self.scene["robot"]
        names = [f"panda_joint{i + 1}" for i in range(7)]
        ids = [robot.data.joint_names.index(n) for n in names]
        limits = torch.as_tensor(self.FR3_PANDA_LIMITS, dtype=torch.float32,
                                 device=robot.device)
        robot.write_joint_position_limit_to_sim(limits.expand(self.num_envs, 7, 2),
                                                joint_ids=ids)
        print("[r2s2r] arm joint limits narrowed to the FR3-and-Panda intersection")

    def _tag_raytraced_subtrees(self):
        """Semantically tag part of the robot so the composite takes the raytraced pixels.

        Same mechanism PolaRiS uses for objects without a splat
        (`manager_based_rl_splat_environment.setup_splat_world_and_robot_views`) and for the
        whole robot when `robot_splat=False` (`droid_cfg.SceneCfg.dynamic_setup`); the only
        difference is that it is applied to a subtree instead of the articulation root.
        """
        stage = get_current_stage()
        for subtree in self.RAYTRACED_SUBTREES:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/robot/{subtree}")
            if not prim.IsValid():
                print(f"[r2s2r] no prim at robot/{subtree}, not tagging")
                continue
            sem = Semantics.SemanticsAPI.Apply(prim, "class_raytraced")
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set("raytraced")
            print(f"[r2s2r] robot/{subtree} drawn by the raytracer")

    def custom_render(self, expensive: bool, transform_static: bool = False):
        rgb = super().custom_render(expensive, transform_static)
        for name in self.WRIST_CAMERAS:
            if name in rgb and rgb[name] is not None and self.wrist_response is not None:
                gain, offset = self.wrist_response
                v = np.asarray(rgb[name]).astype(np.float32) * gain + offset
                rgb[name] = np.clip(v, 0, 255).astype(np.uint8)
        if self.blank_wrist:
            for name in self.WRIST_CAMERAS:
                if name in rgb and rgb[name] is not None:
                    rgb[name] = np.zeros_like(np.asarray(rgb[name]))
        return rgb


# The primary environment is marker free, matching GSWorld's shipped final: ArUcos and the
# dangling arm cables painted out, table-top cables kept (decision of 2026-08-12).
gym.register(
    id="DROID-PourMustard",
    entry_point=PourMustardEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": PourMustardCfg,
        "usd_file": str(DATA_PATH / "pour_mustard/scene.usda"),
        "rubric": pour_mustard_rubric(),
    },
)

# The same environment with the two ArUco markers restored, kept alongside: the real table
# still carries them, so this one is the more faithful render if they are never peeled off
# (measured cost of removing them: wrist PSNR 19.9 -> 17.6 against real frames).
gym.register(
    id="DROID-PourMustardAruco",
    entry_point=PourMustardEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": PourMustardCfg,
        "usd_file": str(DATA_PATH / "pour_mustard_aruco/scene.usda"),
        "rubric": pour_mustard_rubric(),
    },
)
