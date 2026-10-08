"""Offline trajectory synthesis: demo object track -> batched reference joint trajectories.

This is plan §2's "Phase 1 assembly": r2r2r's synthesis organs, taken out of their
isaaclab_viser simulation class and run offline, with no isaacsim dependency. Organ map
(details in provenance):

- state machine and timing: r2r2r `franka_coffee_maker.py` ManipulationConfig /
  ManipulationStateMachine (L27-133). Constants kept: setup 6 steps, gripper closes from
  step 34, follow starts at 38, retract offsets -0.095/-0.005/-0.08 along the grasp
  approach axis, approach interpolation over steps 20..35, lateral EE offset uniform
  +-0.08 m decaying by 0.95^(1/(t-6)) per step. All of these are step counts, so a
  time-scaled episode (below) stretches them by the same factor.
- EE target chain: `_calculate_target_poses` (L718-786):
  flange = obj(t) @ grasp_candidate @ offset @ A^-1, where their Rz(90) @ TCP(0,0,-0.1)
  is the Franka-hand version of A^-1; ours is built from the measured Robotiq geometry
  (tcp.json: grip centre and closing axis in the panda_link8 frame).
- grasp candidates: jaxmp `AntipodalGrasps.to_se3` both flips x rsrd
  `GraspablePart.get_grasp_augs` (8 rolls about the finger axis x 3 translations of
  +-5 mm), the same 48 frames per antipodal pair rsrd feeds r2r2r.
- object trajectory retarget: r2r2r's own chain, unchanged -
  `traj_interp_batch(traj, new_starts, proportion=0.6)` then
  `generate_uniform_control_points_batch(proportion=1.1, tension=0.1)`
  (`franka_coffee_maker.py` L484-496). proportion=0.6 is the part that matters: only the
  leading 60% of the path is bent onto the new start, and the trailing 40% - which is the
  whole pour - stays exactly as demonstrated. Their `generate_directional_starts` is NOT
  used, because it draws a random start while our phase 4 resets the object to
  initial_conditions.json exactly, so the start has to BE the condition's pose; the
  condition file is itself built as a perturbation of the demonstration's layout
  (`make_initial_conditions.py`), which is what makes their interpolation applicable.

  This replaces a `retarget_rigid` of ours (2026-08-12 to 2026-08-13). Its stated reason -
  trajgen's per-dimension affine stretches the path 4-7x out of reach - was real but
  self-inflicted: the conditions then sampled the bottle and the cup independently over
  the table, so half of them asked for a layout mirrored from the demonstration, and no
  interpolation can cover that. With the layout centred on the demonstration the endpoints
  move by centimetres and the official interpolation is in its designed regime.

  The demonstration's table sits ~24 mm above ours (its cup's base is at z 0.0055, our
  table is at -0.019), so the whole demonstrated path is shifted in z once to put it on
  our table before any of this. Without that the object trajectory would start above the
  surface the simulator will have reset the bottle onto.
- batched IK: jaxmp `solve_ik_batched` with their controller's weights (pos 100, rot 5,
  rest 0.01, limit 100, conjugate_gradient, 50 iters), warm started from the previous
  step. The FR3 URDF comes from robot_descriptions (upstream franka_description).

Gates, per episode. r2r2r's three checks - IK position error < 0.02 m, rotation error
< 0.45 rad, joints inside limits - plus the motion budget the real arm enforces and
r2r2r never needed (their data was never executed open loop on hardware):

- joint positions inside the FR3-and-Panda intersection (docs/fr3_limits.md), so the
  trajectory is both executable on the FR3 and replayable in PolaRiS's Panda-limited USD;
- joint velocity under the tightest of every limit on file, per joint and per step
  (`fr3_velocity_window`): FR3's position-dependent envelope from libfranka's
  `rate_limiting.h` (verified verbatim against $REFS, 2026-08-13),
  capped by the constant ceiling, which for this robot is the deployment execution
  layer's soft-safety wall (1.575 rad/s J1-4, 2.01 J5-7), not FR3's catalogue 2.62/5.26/
  4.18. That wall is second-hand - LFHV's transcription of the NUC's
  config/fr3/franka_hardware.yaml, which is not reachable from this machine - and is
  recorded as such in provenance; it is used because it is stricter than every official
  number, so it can only make the gate safer;
- joint acceleration under a ceiling measured on real executed trajectories, not taken
  from the spec. libfranka's 10 rad/s^2 applies to its 1 kHz loop; a 15 Hz waypoint stream
  differenced twice is a different quantity, and `scripts/datagen/measure_motion_budget.py` shows
  the 2026-02-09 teleop episodes - which the real arm ran - reach 20.77 rad/s^2, three of
  eight above 10. Gating at 10 would reject data the robot has demonstrably executed, so
  the ceiling is that measured maximum. The same measurement is what validates the
  velocity gate (real peak 1.39 rad/s, 0.69 of the soft wall) and retires the jerk gate
  (real max 513 against libfranka's 5000), which is reported but not enforced;
- the gripper stays out of the cup. Five proxy points carried by the achieved TCP pose
  must never be inside the cup's cylinder (radius + 1 cm) below the rim + 2 cm - LFHV's
  `cup_violations` (`tools/datagen/gen_kinematic_states.py:230,257`), which they wrote for
  this same task and this same cup. It belongs to the collision family the physics gate is
  for: the first physics rollout had the bottle sweep through the cup and shove it 8.2 cm,
  which no kinematic check on the arm alone would have seen.

An episode that only fails the motion budget is not thrown away: it is re-synthesised
with the whole state machine stretched in time (`--max-time-scale`), which is what the
real arm would do - drive the same path more slowly. Velocity scales as 1/s,
acceleration as 1/s^2, so the retry scale comes straight from the measured overshoot.
The stretch is applied at r2r2r's own knob: every step count in the state machine and
trajgen's `proportion` are multiplied by s, so the episode is the same motion sampled
more finely, not a different motion. Episodes that fail on joint limits or IK are not
retried - slowing down cannot fix either.

Deviations, all recorded: object initial poses come from our initial_conditions.json
instead of their +-6 cm scene randomiser (phase 4 resets the sim to those exact poses, so
the trajectories must be keyed to them); the gripper command is binary 0/1 (polaris's
DROID convention) instead of their per-grasp width; the release phase defaults to off
because the pour demo ends holding the bottle; the home-to-pregrasp ramp is ours (r2r2r
lets an online PD controller chase the first target, an open-loop replay has no
controller to lean on) and follows a smoothstep profile sized to stay inside both the
velocity and the acceleration budget - a linear ramp starts with a velocity step, which
is an infinite acceleration and fails the gate it is supposed to respect.

    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=5 $TRAJ_VENV/bin/python \
        scripts/datagen/synthesize_trajectories.py --conditions 0 1 2 --per-condition 8 \
        --out-dir $WORK/traj/episodes
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

from jaxmp import BatchedRobotFactors, JaxKinTree
from jaxmp.extras.grasp_antipodal import AntipodalGrasps
from jaxmp.extras.solve_ik import solve_ik_batched
from jaxmp.extras.urdf_loader import load_urdf

from trajgen.traj_interp import traj_interp_batch
from trajgen.traj_resampling import generate_uniform_control_points_batch

# the pure-numpy pieces live in the package so they can be tested without the heavy
# dependencies; make the repository's src importable when the package is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# ManipulationConfig defaults, franka_coffee_maker.py L27-67
SETUP_STEPS = 6
GRASP_STEP = 38
RELEASE_OFFSET_STEPS = 20
RETRACT_START = -0.095
RETRACT_GRASP = -0.005
RETRACT_RELEASE = -0.08
APPROACH_INTERP_LEAD = 14                  # L87-93, offset from SETUP_STEPS
APPROACH_INTERP_LAG = 3                    # L87-93, offset back from GRASP_STEP
GRIPPER_CLOSE_LEAD = 4                     # L102-103, offset back from GRASP_STEP
EE_OFFSET_RANGE = 0.08                     # L131-133
EE_OFFSET_DECAY = 0.95                     # L95-96
GRASP_PERTURB_SIGMA = 0.05                 # L415-427, per episode, about grasp x/y
GRASP_PERTURB_STEP_SIGMA = 0.003           # L744-758, per step
TRAJGEN_PROPORTION = 1.1                   # _reset_object_state
TRAJGEN_TENSION = 0.1
TRAJ_INTERP_PROPORTION = 0.6               # franka_coffee_maker.py L489: leading edge only

# Joint limits, the FR3 velocity envelope, the motion budget and the home ramp are the
# pure-numpy part of this script and live in the package (docs/fr3_limits.md explains
# every number). Imported rather than redefined so the environment, the real-robot
# watchdog and the tests all gate on the same arrays.
from polaris_lfhv.limits import (  # noqa: E402
    ACC_MAX, DEPLOY_V_SOFT, ENVELOPE_HIGH, ENVELOPE_LOW, FR3_ACC_MAX, FR3_JERK_MAX,
    FR3_V_A_HI, FR3_V_A_LO, FR3_V_C, FR3_V_K, FR3_V_MAX, HOME, LIMITS_HIGH, LIMITS_LOW,
    RAMP_BUDGET, RATE_HZ, SAFE_HIGH, SAFE_LOW, VEL_CEILING, fr3_velocity_window, home_ramp,
    motion_ratios, smoothstep)

# Gripper stand-in for the cup-clearance check, in the TCP frame (x = closing axis,
# z = approach). Five points: the grip centre, the two finger tips at the half-span, and
# the two knuckles set back along the approach. Same shape and the same numbers as LFHV's
# last round (`tools/datagen/gen_kinematic_states.py:230`), with their y - the finger
# opening direction - mapped onto our x, which is where our measured closing axis lives.
GRIPPER_PROXY = np.array([[0.0, 0.0, 0.0],
                          [0.05, 0.0, 0.0], [-0.05, 0.0, 0.0],
                          [0.06, 0.0, -0.06], [-0.06, 0.0, -0.06]])
# Cup, measured from the mesh by the pour rubric at run time and hardcoded here because
# the synthesiser has no USD stage: radius 0.0468 m, rim 0.1152 m above its base. LFHV
# used 0.047 and 0.115 for the same cup, solved independently.
CUP_RADIUS = 0.0468
CUP_RIM = 0.1152
CUP_MARGIN = 0.01      # their `cup_margin`
CUP_RIM_CLEAR = 0.02   # their allowance above the rim
SCREEN_CHUNK = 128     # grasp candidates per screening solve; 960 at once asks XLA for 60 GB

def detect_quat_order(track):
    """take4.npy stores xyz + a quaternion; which lane is w?

    The cup (object 1) never leaves the table and both meshes stand with their long axis
    +Y up, so the correct reading maps mesh +Y to world +-Z in every frame.
    """
    up = np.array([0.0, 1.0, 0.0])
    best = None
    for name, order in (("wxyz", [0, 1, 2, 3]), ("xyzw", [3, 0, 1, 2])):
        q = track[1, :, 3:][:, order]  # -> wxyz
        rot = jaxlie.SO3(jnp.asarray(q))
        world_up = np.asarray(rot @ jnp.asarray(up))
        score = float(np.abs(world_up[:, 2]).min())
        print(f"quat order {name}: min |up_z| over cup frames = {score:.3f}")
        if score > 0.95:
            best = order
    assert best is not None, "neither quaternion order keeps the cup upright"
    return best


def grasp_candidates(grasps_npz):
    """rsrd GraspablePart.grasp: to_se3 both flips, times get_grasp_augs(8, 3)."""
    g = AntipodalGrasps(centers=jnp.asarray(grasps_npz["centers"]),
                        axes=jnp.asarray(grasps_npz["axes"]))
    base = jaxlie.SE3(jnp.concatenate(
        [g.to_se3().wxyz_xyz, g.to_se3(flip_axis=True).wxyz_xyz], axis=0))
    rot_augs = jaxlie.SE3.from_rotation(
        jaxlie.SO3.from_x_radians(jnp.linspace(-jnp.pi, jnp.pi, 8)))
    trans_augs = jaxlie.SE3.from_translation(
        jnp.array([[d, 0.0, 0.0] for d in np.linspace(-0.005, 0.005, 3)]))
    augs = jaxlie.SE3(jnp.repeat(trans_augs.wxyz_xyz, 8, axis=0)).multiply(
        jaxlie.SE3(jnp.tile(rot_augs.wxyz_xyz, (3, 1))))
    cand = jaxlie.SE3(base.wxyz_xyz[:, None, :]).multiply(
        jaxlie.SE3(augs.wxyz_xyz[None, :, :]))
    return jaxlie.SE3(cand.wxyz_xyz.reshape(-1, 7))


def flange_from_grasp(tcp):
    """A^-1: candidate grasp frame -> flange, from the measured Robotiq geometry.

    The grasp frame has x along the finger-closing axis and z along the approach; in the
    flange frame the measured closing axis and the approach (flange z) define the same
    frame at the grip centre. Their Franka version is Rz(90) @ (0,0,-0.1034)."""
    x = np.asarray(tcp["closing_axis_in_link8"], float)
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    z -= z.dot(x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    r = np.stack([x, y, z], axis=1)
    a = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(jnp.asarray(r)), jnp.asarray(tcp["tcp_in_link8"]))
    return a.inverse()


class Synthesiser:
    """Everything that does not depend on the initial condition, loaded once."""

    def __init__(self, args):
        track = np.load(args.track)          # (2 objects, T, xyz + quat)
        order = detect_quat_order(track)
        track = np.concatenate([track[..., :3], track[..., 3:][..., order]], axis=-1)
        self.demo_bottle = track[0].copy()   # (T, xyz + wxyz)
        self.demo_cup_xyz = track[1].mean(0)[:3]
        # the demonstration's table is not ours: its cup stands on z 0.0055 while our
        # measured tabletop is -0.019 (docs/gsworld_to_polaris.md). Put the demonstrated
        # path on our table once, here, so the trajectory starts on the surface the
        # simulator will have reset the bottle onto. Only z moves; the lift profile above
        # the table is the demonstration's.
        self.z_shift = float(args.table_z - self.demo_cup_xyz[2])
        self.demo_bottle[:, 2] += self.z_shift
        print(f"track: {track.shape[1]} frames; demo table z {self.demo_cup_xyz[2]:.4f} -> "
              f"ours {args.table_z:.4f} (shift {self.z_shift:+.4f} m); bottle lifts to "
              f"{float(self.demo_bottle[:, 2].max() - args.table_z):.3f} m above the table")
        self.demo_cup_xy = self.demo_cup_xyz[:2]
        self.demo_cup_spin = self.cup_spin(track[1, 0, 3:])
        self.cup_tol = args.cup_tolerance

        self.cands = grasp_candidates(np.load(args.grasps))
        print(f"{self.cands.wxyz_xyz.shape[0]} grasp candidates")
        self.a_inv = flange_from_grasp(json.load(open(args.tcp)))
        self.a = self.a_inv.inverse()        # flange -> TCP, for the cup clearance check

        # The axis the demonstration tilts the bottle about, in the bottle's own frame.
        # LFHV chose the pour grasp so its closing axis is parallel to this ("alignment
        # 1.00, so pouring is roughly a spin about the held axis"); anything else pours by
        # swinging the arm. Measured here rather than copied: 86.9 degrees about
        # (0.003, -0.150, -0.989) in the bottle frame, i.e. across the bottle, perpendicular
        # to its long axis.
        from scipy.spatial.transform import Rotation
        rb = Rotation.from_quat(np.roll(track[0, :, 3:], -1, axis=1))
        rv = (rb[-1] * rb[0].inv()).as_rotvec()
        self.tilt_axis = rb[0].inv().apply(rv / np.linalg.norm(rv))
        self.grasp_rot0 = rb[0]
        print(f"demo tilt: {np.degrees(np.linalg.norm(rv)):.1f} deg about "
              f"{np.round(self.tilt_axis, 3)} in the bottle's frame")
        self.align_min = args.grasp_align
        self.approach_deg = args.approach_deg
        self.release = args.release

        urdf = load_urdf("fr3_description")
        self.kin = JaxKinTree.from_urdf(urdf)
        self.target_idx = jnp.array([self.kin.joint_names.index("fr3_joint8")])
        self.rest = jnp.concatenate(
            [jnp.asarray(HOME), jnp.zeros(self.kin.num_actuated_joints - 7)])
        self.ik_weight = jnp.array([100.0] * 3 + [5.0] * 3)

    def world_trajectory(self, cond, scale):
        """The bottle's world trajectory for this condition, resampled at `scale` x time.

        r2r2r's chain verbatim (`franka_coffee_maker.py` L486-496): bend the leading 60%
        onto the new start, leave the trailing 40% - the pour - exactly as demonstrated,
        then resample. The start is the condition's own bottle pose rather than one of
        their random `generate_directional_starts` draws, because phase 4 resets the object
        to that pose.

        This only works because the cup does not move: the pour is the demonstrated one, in
        the demonstrated place. Conditions come from `make_initial_conditions.py`, which
        follows LFHV's last round and keeps the cup at its demo pose while perturbing the
        bottle. A moved cup would need the trailing edge retargeted too, which is a
        different problem - so it is refused rather than silently mishandled.
        """
        cup_xy = np.array(cond["blue_cup"], float)[:2]
        moved = float(np.linalg.norm(cup_xy - self.demo_cup_xy))
        if moved > self.cup_tol:
            raise SystemExit(
                f"the cup is {moved * 100:.1f} cm from the demonstration's, past the "
                f"--cup-tolerance of {self.cup_tol * 100:.1f} cm. The pour is retargeted "
                f"onto the cup by interpolating both endpoints, which only behaves for "
                f"small displacements; regenerate the conditions with a smaller --cup-xy.")

        bottle_init = np.array(cond["mustard"], float)     # x y z qw qx qy qz
        new_start = np.concatenate([bottle_init[[3, 4, 5, 6]], bottle_init[:3]])[None]
        traj = np.concatenate([self.demo_bottle[:, 3:], self.demo_bottle[:, :3]], axis=1)
        # The cup has a small draw of its own, so the pour has to follow it. Do that by
        # carrying the WHOLE demonstrated path rigidly onto this cup first - turn it about
        # the cup's centre by however much the cup turned, then translate by however much
        # it moved - and only then run r2r2r's proportion=0.6 retarget onto the condition's
        # bottle. The trailing 40% therefore lands on this cup by construction, and the
        # leading edge is bent exactly as it is when the cup does not move.
        #
        # The alternative, letting trajgen interpolate both endpoints (new_ends with
        # proportion=1.0), was tried on 2026-08-13 and is wrong here: it warps the leading
        # edge too, and the grasp then lands 4.2-4.6 cm off the bottle instead of 0.5-1.3,
        # which loses the grasp. Interpolating an endpoint is for moving an endpoint, not
        # for moving the whole scene.
        traj = self.carry_to_cup(traj, cond)
        interp = traj_interp_batch(traj=traj, new_starts=new_start,
                                   proportion=TRAJ_INTERP_PROPORTION)
        return generate_uniform_control_points_batch(
            interp, proportion=TRAJGEN_PROPORTION * scale,
            tension=TRAJGEN_TENSION)[0]                    # (T', wxyz + xyz)

    def carry_to_cup(self, traj, cond):
        """Move a whole (T, wxyz+xyz) path rigidly onto this condition's cup.

        Turn about the cup's vertical axis by however much the cup turned, then translate
        by however much it moved. The path's shape is untouched, so the pour stays the
        demonstrated pour - it just happens where this cup is.
        """
        from scipy.spatial.transform import Rotation

        cup = np.array(cond["blue_cup"], float)
        dxy = cup[:2] - self.demo_cup_xy
        dyaw = self.cup_spin(cup[3:]) - self.demo_cup_spin
        if abs(dyaw) < 1e-6 and np.linalg.norm(dxy) < 1e-6:
            return traj
        yaw = Rotation.from_euler("z", dyaw)
        centre = np.array([self.demo_cup_xy[0], self.demo_cup_xy[1], 0.0])
        pos = traj[:, 4:].copy()
        pos = centre + yaw.apply(pos - centre) + np.array([dxy[0], dxy[1], 0.0])
        rot = yaw * Rotation.from_quat(np.roll(traj[:, :4], -1, axis=1))
        return np.concatenate([np.roll(rot.as_quat(), 1, axis=1), pos], axis=1)

    @staticmethod
    def cup_spin(quat_wxyz):
        """Yaw of a standing object about world z, from its wxyz quaternion."""
        from scipy.spatial.transform import Rotation
        f = Rotation.from_quat(np.roll(np.asarray(quat_wxyz, float), -1)).apply([0, 0, 1])
        return float(np.arctan2(f[1], f[0]))

    def grasp_shape_ok(self):
        """Keep only the candidates that grasp the way the demonstration is poured from.

        Two shape criteria, both taken from LFHV's last round, which states them for this
        exact task and cup (`tools/datagen/gen_kinematic_states.py:71-77`): "The pour task
        uses a side grasp. A top grasp puts the palm over the spout and needs a large arm
        excursion to tilt. ... its closing axis is parallel to the demo's tilt axis
        (alignment 1.00, so pouring is roughly a spin about the held axis), it approaches
        horizontally from +y". They picked one candidate by hand in a GUI; the same two
        properties are measurable, so here they are applied to all 960 instead.

        Measured on their generated qpos for comparison (82 episodes, FR3 forward
        kinematics): the gripper approaches 23-29 degrees below horizontal, the palm ends
        fully down (flange +z dot world -z = 0.98-1.00), and the last quarter of the
        episode is carried by j5 (65-74 deg) and j7 (66-67 deg) while j1/j2/j4 move 1-6.
        Without these criteria ours came out at 56 degrees below horizontal - a top grasp -
        and poured by swinging j1/j2/j4 through 27-36 degrees, which is the excursion their
        note warns about and what drove the bottle through the cup.
        """
        M = np.asarray(self.cands.rotation().as_matrix())    # columns: closing, y, approach
        align = np.abs(M[:, :, 0] @ self.tilt_axis)
        # the approach in the world with the bottle standing as the demonstration left it
        down = -self.grasp_rot0.apply(M[:, :, 2])[:, 2]
        lo, hi = np.sin(np.radians(self.approach_deg))
        ok = (align >= self.align_min) & (down >= lo) & (down <= hi)
        print(f"  grasp shape: {int(ok.sum())}/{len(ok)} candidates have closing axis "
              f"aligned >= {self.align_min} and approach "
              f"{self.approach_deg[0]:.0f}-{self.approach_deg[1]:.0f} deg below horizontal")
        return ok

    def screen_grasps(self, cond, margin=0.10):
        """Which grasp candidates can this condition's key poses be reached with at all?

        r2r2r draws a grasp at random per episode and lets the kinematic checks throw the
        bad ones away, which is also what plan §3 says to do - "which grasp to use is left
        to the kinematic checks and the physics gate". At their end poses that is cheap;
        at ours it is not, because the pour turns the flange about 84 degrees off upright
        (measured on the demonstration) and most of the 960 candidates ask for a wrist the
        FR3 does not have. Screening first turns a 7% episode yield into a usable one and
        changes nothing about the criterion - it applies the same joint limits, at three
        poses of the trajectory instead of all of them.

        The margin is for selection only, not a gate: a candidate that reaches a key pose
        exactly at a limit has no room for the transport in between. Limits are the FR3-
        and-Panda intersection further clipped to where FR3's velocity envelope still
        allows motion, since a pose past that is unusable however slowly it is approached.
        """
        world = self.world_trajectory(cond, 1.0)
        keys = [world[0], world[len(world) // 2], world[-1]]
        ok = self.grasp_shape_ok()
        # Only the shape-passing candidates are worth an IK solve. They are a small
        # fraction (126 of 2880 at the settled thresholds), and the shape test is pure
        # numpy on the candidate frames, so screening the whole pool costs 20x more
        # solves for no extra survivors.
        idx = np.flatnonzero(ok)
        # chunked: one solve over 960 candidates at once asks XLA for 60 GB
        for lo in range(0, len(idx), SCREEN_CHUNK):
            sub = idx[lo:lo + SCREEN_CHUNK]
            cands = jaxlie.SE3(self.cands.wxyz_xyz[jnp.asarray(sub)])
            n = cands.wxyz_xyz.shape[0]
            JointVar = BatchedRobotFactors.get_var_class(self.kin, self.rest, n)
            offset = jaxlie.SE3.from_translation(
                jnp.tile(jnp.array([0.0, 0.0, RETRACT_GRASP]), (n, 1)))
            joints_prev = jnp.tile(self.rest[None], (n, 1))
            for w in keys:
                obj = jaxlie.SE3(jnp.tile(jnp.asarray(w)[None], (n, 1)))
                flange = obj.multiply(cands).multiply(offset).multiply(self.a_inv)
                _, joints = solve_ik_batched(
                    self.kin, jaxlie.SE3(flange.wxyz_xyz[:, None, :]), self.target_idx,
                    joints_prev, JointVar, self.ik_weight, num_batches=n)
                joints_prev = joints
                q = np.asarray(joints[:, :7])
                fk = jaxlie.SE3(self.kin.forward_kinematics(joints)[:, self.target_idx[0]])
                pos_err = np.linalg.norm(
                    np.asarray(fk.translation()) - np.asarray(flange.translation()), axis=1)
                rot_err = np.abs(np.asarray(
                    (fk.rotation().inverse() @ flange.rotation()).log())).max(axis=1)
                ok[sub] &= (pos_err < 0.02) & (rot_err < 0.45)
                ok[sub] &= ((q >= SAFE_LOW + margin) & (q <= SAFE_HIGH - margin)).all(axis=1)
        return np.flatnonzero(ok)

    def run(self, cond, draws, scale, rng):
        """Synthesise one batch of episodes for one condition at one time scale.

        `draws` holds the per-episode randomisation (grasp frames and lateral offsets), so
        a retry at a larger scale is the same motion driven more slowly, not a new one.
        """
        b = draws["grasps"].wxyz_xyz.shape[0]
        setup = max(1, int(round(SETUP_STEPS * scale)))
        grasp_step = int(round(GRASP_STEP * scale))
        interp_start = setup + int(round(APPROACH_INTERP_LEAD * scale))
        interp_end = grasp_step - int(round(APPROACH_INTERP_LAG * scale))
        close_step = grasp_step - int(round(GRIPPER_CLOSE_LEAD * scale))
        decay = EE_OFFSET_DECAY ** (1.0 / scale)   # same total decay over a longer approach

        world_traj = self.world_trajectory(cond, scale)
        t_follow = world_traj.shape[0]
        release_step = grasp_step + t_follow
        total = release_step + (int(round(RELEASE_OFFSET_STEPS * scale)) if self.release else 3)

        qpos = np.zeros((total, b, 7))
        grip = np.zeros((total, b))
        obj_traj = np.zeros((total, b, 7))  # xyz + wxyz, mustard
        ik_pos_err = np.zeros((total, b))
        ik_rot_err = np.zeros((total, b))
        cup_hits = np.zeros((total, b), bool)
        cup_xy = np.array(cond["blue_cup"], float)[:2]
        cup_rim = float(np.array(cond["blue_cup"], float)[2]) + CUP_RIM
        joints_prev = jnp.tile(self.rest[None], (b, 1))
        JointVar = BatchedRobotFactors.get_var_class(self.kin, self.rest, b)

        for t in range(total):
            # object pose this step (L524-571): hold the condition pose until grasp,
            # then follow the retargeted trajectory
            k = int(np.clip(t - grasp_step, 0, t_follow - 1))
            obj_t = jaxlie.SE3(jnp.tile(jnp.asarray(world_traj[k])[None], (b, 1)))

            # retract along the approach axis (state machine L78-116)
            if t <= setup:
                dz, lateral_scale = RETRACT_START, 1.0
            elif t < grasp_step:
                s = smoothstep((t - interp_start) / max(1, interp_end - interp_start))
                dz = RETRACT_START + (RETRACT_GRASP - RETRACT_START) * s
                lateral_scale = decay ** (t - setup)
            elif t < release_step:
                dz, lateral_scale = RETRACT_GRASP, 0.0
            else:
                s = smoothstep((t - release_step - 5) / max(1, total - release_step - 5))
                dz = RETRACT_GRASP + (RETRACT_RELEASE - RETRACT_GRASP) * s \
                    if self.release else RETRACT_GRASP
                lateral_scale = 0.0

            lat = draws["ee_offset"] * lateral_scale * (abs(dz) / abs(RETRACT_START))
            offset = jaxlie.SE3.from_translation(jnp.concatenate(
                [jnp.asarray(lat), jnp.full((b, 1), dz)], axis=1))
            step_pert = jaxlie.SE3.from_rotation(jaxlie.SO3.from_rpy_radians(
                jnp.asarray(rng.normal(0, GRASP_PERTURB_STEP_SIGMA, size=b)),
                jnp.asarray(rng.normal(0, GRASP_PERTURB_STEP_SIGMA, size=b)),
                jnp.zeros(b)))
            flange = obj_t.multiply(draws["grasps"]).multiply(step_pert) \
                          .multiply(offset).multiply(self.a_inv)

            sol_pose, joints = solve_ik_batched(
                self.kin, jaxlie.SE3(flange.wxyz_xyz[:, None, :]), self.target_idx,
                joints_prev, JointVar, self.ik_weight, num_batches=b)
            joints_prev = joints
            qpos[t] = np.asarray(joints[:, :7])
            grip[t] = 1.0 if (close_step <= t < (release_step if self.release
                                                 else total)) else 0.0
            obj_traj[t] = np.concatenate(
                [np.asarray(obj_t.translation()),
                 np.asarray(obj_t.rotation().wxyz)], axis=1)

            fk = self.kin.forward_kinematics(joints)[:, self.target_idx[0]]
            fk_pose = jaxlie.SE3(fk)
            ik_pos_err[t] = np.linalg.norm(
                np.asarray(fk_pose.translation()) - np.asarray(flange.translation()), axis=1)
            ik_rot_err[t] = np.abs(np.asarray(
                (fk_pose.rotation().inverse() @ flange.rotation()).log())).max(axis=1)

            # gripper against the cup: five proxy points carried by the achieved TCP pose
            tcp = fk_pose.multiply(jaxlie.SE3(jnp.tile(self.a.wxyz_xyz[None], (b, 1))))
            rot = np.asarray(tcp.rotation().as_matrix())      # (b, 3, 3)
            pts = (np.asarray(tcp.translation())[:, None, :]
                   + np.einsum("bij,kj->bki", rot, GRIPPER_PROXY))   # (b, 5, 3)
            horiz = np.linalg.norm(pts[:, :, :2] - cup_xy[None, None, :], axis=-1)
            cup_hits[t] = ((horiz < CUP_RADIUS + CUP_MARGIN)
                           & (pts[:, :, 2] < cup_rim + CUP_RIM_CLEAR)).any(axis=1)

        ramp = home_ramp(qpos[0])
        n_ramp = ramp.shape[0]
        pad = np.zeros((n_ramp, b))
        return {
            "qpos": np.concatenate([ramp, qpos], axis=0),
            "gripper": np.concatenate([pad, grip], axis=0),
            "obj": np.concatenate([np.tile(obj_traj[:1], (n_ramp, 1, 1)), obj_traj], axis=0),
            "ik_pos": np.concatenate([pad, ik_pos_err], axis=0),
            "ik_rot": np.concatenate([pad, ik_rot_err], axis=0),
            "cup_hits": cup_hits.any(axis=0),
            "n_ramp": n_ramp, "setup": setup, "close_step": n_ramp + close_step,
            "grasp_step": n_ramp + grasp_step, "t_follow": t_follow, "scale": scale,
        }


def worst_step(out, rate=RATE_HZ):
    """Where the velocity budget is worst, for reading a failing batch."""
    qpos = out["qpos"]
    dq = np.diff(qpos, axis=0) * rate
    lo, hi = fr3_velocity_window(qpos[:-1])
    ratio = np.where(dq >= 0, dq / np.maximum(hi, 1e-9), dq / np.minimum(lo, -1e-9))
    t, ep, j = np.unravel_index(np.argmax(ratio), ratio.shape)
    if t < out["n_ramp"]:
        phase = "ramp"
    elif t < out["grasp_step"]:
        phase = "approach"
    else:
        phase = "follow"
    return {"step": int(t), "joint": int(j), "ratio": float(ratio[t, ep, j]), "phase": phase}


def gate(out):
    """r2r2r's three checks plus the motion budget. Returns per-episode flags and ratios."""
    qpos = out["qpos"]
    follow = slice(out["n_ramp"] + out["setup"] + 1, None)
    vel, acc, jerk = motion_ratios(qpos)
    return {
        "limits": ((qpos >= LIMITS_LOW) & (qpos <= LIMITS_HIGH)).all(axis=(0, 2)),
        "vel": vel <= 1.0, "acc": acc <= 1.0,
        "ik_pos": (out["ik_pos"][follow] < 0.02).all(axis=0),
        "ik_rot": (out["ik_rot"][follow] < 0.45).all(axis=0),
        "cup_clear": ~out["cup_hits"],
        "vel_ratio": vel, "acc_ratio": acc, "jerk_ratio": jerk,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default=os.environ.get("DEMO_TRACK", "take4.npy"))
    p.add_argument("--grasps", default=os.path.join(os.environ.get("WORK", "work"), "traj/mustard_grasps.npz"))
    p.add_argument("--tcp", default=os.path.join(os.environ.get("WORK", "work"), "traj/tcp.json"))
    p.add_argument("--conditions-json",
                   default=os.path.join(os.environ.get("POLARIS_ROOT", "polaris"), "PolaRiS-Hub/pour_mustard/initial_conditions.json"))
    p.add_argument("--conditions", type=int, nargs="*", default=None,
                   help="condition indices; default all")
    p.add_argument("--per-condition", type=int, default=8, help="episodes per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table-z", type=float, default=-0.019,
                   help="our tabletop, where the conditions put an object's base; the "
                        "demonstrated path is shifted in z onto it")
    p.add_argument("--grasp-align", type=float, default=0.90,
                   help="minimum |closing axis . demo tilt axis|; LFHV's hand-picked "
                        "candidate is 1.00, so the pour is a spin about the held axis")
    p.add_argument("--approach-deg", type=float, nargs=2, default=[10.0, 40.0],
                   help="how far below horizontal the gripper may approach. A side grasp; "
                        "LFHV's generated episodes measure 23-29 deg, a top grasp is 56")
    p.add_argument("--cup-tolerance", type=float, default=0.08,
                   help="how far the condition's cup may sit from the demonstration's "
                        "before the trailing edge of the path stops being valid")
    p.add_argument("--max-time-scale", type=float, default=4.0,
                   help="largest time stretch a motion-budget retry may use; 1 disables retries")
    p.add_argument("--retry-rounds", type=int, default=2)
    p.add_argument("--release", action="store_true",
                   help="open the gripper and retract at the end (their default; the pour "
                        "demo ends holding the bottle, so ours is off)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", default=None, help="where to write the per-condition gate table")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn = Synthesiser(args)
    conditions = json.load(open(args.conditions_json))["poses"]
    picked = args.conditions if args.conditions is not None else list(range(len(conditions)))

    index, table = [], []
    for cond_id in picked:
        b = args.per_condition
        cond = conditions[cond_id]

        pool = syn.screen_grasps(cond)
        if len(pool) == 0:
            print(f"condition {cond_id}: no grasp candidate reaches all three key poses, "
                  f"skipped")
            table.append({"condition": cond_id, "passed": 0, "tried": 0, "pool": 0})
            continue
        print(f"  cond {cond_id}: {len(pool)}/{syn.cands.wxyz_xyz.shape[0]} grasp "
              f"candidates reach grasp, mid-transport and pour")

        # per-episode draws (their reset-time randomisation), fixed across retries
        grasp_ids = pool[rng.integers(0, len(pool), size=b)]
        ep_grasps = jaxlie.SE3(syn.cands.wxyz_xyz[jnp.asarray(grasp_ids)])
        perturb = jaxlie.SO3.from_rpy_radians(
            jnp.asarray(rng.normal(0, GRASP_PERTURB_SIGMA, size=b)),
            jnp.asarray(rng.normal(0, GRASP_PERTURB_SIGMA, size=b)),
            jnp.zeros(b))
        all_draws = {"grasps": ep_grasps.multiply(jaxlie.SE3.from_rotation(perturb)),
                     "ee_offset": rng.uniform(-EE_OFFSET_RANGE, EE_OFFSET_RANGE, size=(b, 2))}

        kept = {}                       # episode index -> (out, slot, scale)
        alive = np.arange(b)            # episodes still worth trying
        scale = 1.0
        stats = {"limits": 0, "vel": 0, "acc": 0, "ik": 0, "rescued": 0}
        for rnd in range(args.retry_rounds + 1):
            draws = {"grasps": jaxlie.SE3(all_draws["grasps"].wxyz_xyz[jnp.asarray(alive)]),
                     "ee_offset": all_draws["ee_offset"][alive]}
            out = syn.run(cond, draws, scale, np.random.default_rng(args.seed + cond_id))
            g = gate(out)
            passed = (g["limits"] & g["vel"] & g["acc"] & g["ik_pos"] & g["ik_rot"]
                      & g["cup_clear"])
            for slot, ep in enumerate(alive):
                if passed[slot]:
                    kept[int(ep)] = (out, slot, scale)
                    if rnd > 0:
                        stats["rescued"] += 1
            if rnd == 0:
                stats["limits"] = int((~g["limits"]).sum())
                stats["vel"] = int((~(g["vel"] & g["acc"])).sum())
                stats["ik"] = int((~(g["ik_pos"] & g["ik_rot"])).sum())
                stats["cup"] = int((~g["cup_clear"]).sum())
                stats["vel_ratio_med"] = float(np.median(g["vel_ratio"]))
                stats["acc_ratio_med"] = float(np.median(g["acc_ratio"]))
                stats["jerk_ratio_med"] = float(np.median(g["jerk_ratio"]))
                print(f"  cond {cond_id} first pass ratios (median over {b}): "
                      f"vel {stats['vel_ratio_med']:.2f}, acc {stats['acc_ratio_med']:.2f}, "
                      f"jerk {stats['jerk_ratio_med']:.3f}; "
                      f"worst-episode vel {g['vel_ratio'].max():.2f} acc {g['acc_ratio'].max():.2f}")
                where = worst_step(out)
                print(f"  peak velocity at step {where['step']}/{out['qpos'].shape[0]} "
                      f"joint j{where['joint'] + 1} ({where['phase']}), "
                      f"ratio {where['ratio']:.2f}")

            # only a motion-budget failure can be fixed by slowing down
            retryable = (~passed) & g["limits"] & g["ik_pos"] & g["ik_rot"] & g["cup_clear"]
            need = np.maximum(g["vel_ratio"], np.sqrt(np.maximum(g["acc_ratio"], 0)))
            retryable &= need * scale <= args.max_time_scale
            if not retryable.any() or scale >= args.max_time_scale:
                break
            scale = min(args.max_time_scale,
                        float(np.ceil(np.max(need[retryable]) * scale * 1.05 * 4) / 4))
            alive = alive[retryable]

        for ep in sorted(kept):
            out, slot, sc = kept[ep]
            name = f"cond{cond_id:03d}_ep{ep:02d}"
            np.savez(out_dir / f"{name}.npz",
                     qpos=out["qpos"][:, slot], gripper=out["gripper"][:, slot],
                     mustard_traj=out["obj"][:, slot],
                     blue_cup_pose=np.array(cond["blue_cup"], float),
                     condition_id=cond_id, grasp_id=grasp_ids[ep],
                     rate_hz=RATE_HZ, n_ramp=out["n_ramp"], time_scale=sc,
                     close_step=out["close_step"], grasp_step=out["grasp_step"],
                     ik_pos_err_max=float(out["ik_pos"][:, slot].max()),
                     ik_rot_err_max=float(out["ik_rot"][:, slot].max()))
            index.append({"episode": name, "condition": cond_id, "time_scale": sc,
                          "steps": int(out["qpos"].shape[0]), "grasp": int(grasp_ids[ep])})
        row = {"condition": cond_id, "passed": len(kept), "tried": b, "pool": len(pool),
               **stats, "final_scale": scale}
        table.append(row)
        print(f"condition {cond_id}: {len(kept)}/{b} passed "
              f"(first pass failed: limits {stats['limits']}, motion {stats['vel']}, "
              f"ik {stats['ik']}, cup {stats.get('cup', 0)}; "
              f"rescued by slowing down {stats['rescued']}, "
              f"largest scale {scale:g})")

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(table, f, indent=1)
    total_tried = sum(r["tried"] for r in table)
    print(f"wrote {len(index)} episodes to {out_dir} "
          f"({len(index)}/{total_tried} = {100 * len(index) / max(1, total_tried):.0f}% "
          f"through the gate, {sum(r.get('rescued', 0) for r in table)} of them rescued)")


if __name__ == "__main__":
    main()
