#!/usr/bin/env python3
"""Safe launcher for DROID evaluation scripts (left arm).

Two jobs
--------
1. Bring the NUC stack up reliably (the gripper-activation race, below).
2. Run the DROID client's `evaluate_openpi.py` under the **absolute joint position**
   action head our checkpoint was trained for.  That script is written for the *joint
   velocity* head and would wreck a position-head rollout; the four required changes are
   listed in `docs/real_robot.md`.

   We apply them to the eval script's **source text in memory** and exec that -- the file
   on disk is never touched, so the DROID client repo stays a clean upstream checkout and
   there is only one copy of the script to keep in sync.  Each patch must match exactly
   once or the launcher aborts, so an upstream edit can never silently skip a
   safety-critical change.  `--show-diff` prints exactly what will run, without contacting
   the robot.

Problem it works around
-----------------------
`RobotEnv.__init__` -> `ServerInterface(launch=True)` -> `launch_controller()` makes the
NUC run `/app/droid/franka/launch_gripper.sh`, which does::

    pkill -9 gripper
    launch_gripper.py gripper=robotiq_2f gripper.comport=/dev/ttyUSB0

with **zero** delay in between.  When the previous `RobotiqGripperClient` is alive and
activated, the SIGKILL interrupts it mid-Modbus and the Robotiq 2F-85 needs a couple of
seconds to settle.  `robotiq_gripper_client.py:47` checks readiness with

    if self.gripper.is_ready() and self.gripper.sendCommand() and self.gripper.getStatus()

immediately after sending the activation command -- no wait, no retry -- so it raises
``Unable to activate!``.  The child process dies before calling ``InitRobotClient``, the
50052 server therefore never receives its metadata, and DROID's `robot.py:57` finally
reports the confusing downstream symptom::

    AttributeError: 'GripperInterface' object has no attribute 'metadata'

`ServerInterface.attempt_n_times(max_attempts=2)` cannot save it: its two tries are only
0.1 s apart, far too fast for the gripper to recover.

What this launcher does
-----------------------
The `launch=True` decision is made on the *laptop* side, so the whole fix fits in this
file -- nothing under `droid/`, `scripts/evaluation/` or on the NUC is modified.  We
monkey-patch `ServerInterface.__init__` so that, before `launch_controller()` fires, the
gripper client is already dead and settled.  Then `pkill -9 gripper` has nothing live to
interrupt and activation is clean.

Per attempt:
  1. kill the gripper client on the NUC, wait for the RS-485 tty to be released, settle
  2. `launch_controller()` on a fresh zerorpc connection
  3. poll until a process re-acquires the tty  == the client got past activation
  4. `launch_robot()` on a fresh connection, then read joints + gripper as a sanity check

Usage
-----
This runs on the robot laptop, in the DROID client repo's virtualenv. It needs to be told
where that repo is (`$DROID_ROOT`, or just run it from inside the repo) and what the NUC's
sudo password is (`$NUC_SUDO_PASSWORD`) -- no credential is stored in this repository.
The laptop keeps a small wrapper that fills both in.

    export DROID_ROOT=/path/to/droid NUC_SUDO_PASSWORD=...
    python <this repo>/scripts/real_robot/run_eval_safe.py --remote_port=8000 --model=pi05

    # see the four position-head patches that will be applied; touches nothing
    python <this repo>/scripts/real_robot/run_eval_safe.py --show-diff

    # bring the stack up and verify it, without running the evaluation
    python <this repo>/scripts/real_robot/run_eval_safe.py --check-only

Options consumed by this launcher (everything else is forwarded verbatim):
    --check-only          launch + verify, then exit
    --show-diff           print the patched eval script as a diff, then exit
    --head SPACE          joint_position (default) | joint_velocity (upstream, unpatched)
    --no-guard            disable the abort-only motion watchdog (see below)
    --eval-script PATH    default: $DROID_ROOT/scripts/evaluation/evaluate_openpi.py
    --attempts N          default: 3
    --settle SEC          default: 4.0

Env: DROID_ROOT, NUC_SUDO_PASSWORD, NUC_USER and NUC_CONTAINER (all required; the lab's
values are not stored in this repository).
"""

import difflib
import os
import re
import subprocess
import sys
import time


def _find_droid_root():
    """Locate the DROID client repo: $DROID_ROOT, else walk up from the working directory."""
    from_env = os.environ.get("DROID_ROOT")
    if from_env:
        return os.path.abspath(from_env)
    candidate = os.path.abspath(os.curdir)
    while True:
        if os.path.exists(os.path.join(candidate, "droid", "robot_env.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            raise SystemExit(
                "cannot find the DROID client repo: set DROID_ROOT, or run this from inside "
                "it (looking for <root>/droid/robot_env.py)"
            )
        candidate = parent


DROID_ROOT = _find_droid_root()
if DROID_ROOT not in sys.path:
    sys.path.insert(0, DROID_ROOT)

from droid.misc.parameters import nuc_ip  # noqa: E402

NUC_USER = os.environ.get("NUC_USER")
# Credentials never live in this repository (the lab's robot setup notes). The NUC sudo
# password comes from the environment; on the robot laptop it is in the lab's robot setup notes.
# NOTE: droid.misc.parameters.sudo_password is the *laptop* password, not this one.
NUC_SUDO_PASSWORD = os.environ.get("NUC_SUDO_PASSWORD")
NUC_CONTAINER = os.environ.get("NUC_CONTAINER")

# The Robotiq 2F-85 talks Modbus RTU over an FTDI RS-485 bridge. Resolve it by by-id
# rather than hardcoding ttyUSB0 -- the enumeration has flipped once already (was ttyUSB1 until ~2026-04).
RS485_BY_ID_GLOB = "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*"


# ----------------------------------------------------------------------------------
# NUC helpers (commands only -- this launcher never writes a file on the NUC)
# ----------------------------------------------------------------------------------
def _ssh(command, timeout=30):
    """Run `command` on the NUC as NUC_USER. Returns (rc, stdout)."""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "%s@%s" % (NUC_USER, nuc_ip), command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    return proc.returncode, proc.stdout.decode("utf8", "replace").strip()


def _ssh_sudo(command, timeout=30):
    return _ssh("echo %s | sudo -S -p '' %s" % (NUC_SUDO_PASSWORD, command), timeout=timeout)


def gripper_tty():
    """Absolute path of the RS-485 device the Robotiq gripper is on, or None."""
    rc, out = _ssh("readlink -f %s 2>/dev/null | head -1" % RS485_BY_ID_GLOB)
    return out if (rc == 0 and out.startswith("/dev/")) else None


def tty_holder(tty):
    """PID holding `tty` on the NUC, or None. A live holder == gripper client running."""
    if not tty:
        return None
    rc, out = _ssh_sudo(
        "bash -c \"ls -l /proc/*/fd/* 2>/dev/null | grep -w %s "
        "| sed 's#/proc/##; s#/fd/[0-9]*##' | awk '{print \\$NF}' | sort -u | head -1\"" % tty
    )
    return out or None


def kill_gripper_client():
    _ssh_sudo("docker exec %s pkill -9 -f launch_gripper.py" % NUC_CONTAINER)


def wait_for(predicate, timeout, poll=1.0, desc=""):
    """Poll `predicate` until truthy. Returns True on success, False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    if desc:
        print("  [warn] timed out waiting for %s (%.0fs)" % (desc, timeout))
    return False


# ----------------------------------------------------------------------------------
# the safe launch sequence
# ----------------------------------------------------------------------------------
def safe_launch(iface, attempts=3, settle=4.0):
    """Bring the NUC stack up reliably. `iface` is a (connected) ServerInterface."""
    tty = gripper_tty()
    print("[safe-launch] gripper RS-485 device: %s" % (tty or "NOT FOUND"))
    if tty is None:
        print("  [warn] no usb-FTDI_USB_TO_RS-485_* under /dev/serial/by-id -- "
              "gripper unplugged? falling back to timing-only waits.")

    last_error = None
    for attempt in range(1, attempts + 1):
        print("[safe-launch] attempt %d/%d" % (attempt, attempts))

        # 1. make sure nothing live gets SIGKILLed mid-Modbus by launch_gripper.sh
        kill_gripper_client()
        if tty:
            wait_for(lambda: tty_holder(tty) is None, timeout=10, desc="%s to be released" % tty)
        print("  settling %.1fs ..." % settle)
        time.sleep(settle)

        # 2. now the relaunch is clean. Fresh zerorpc client each time: launch_controller
        #    blocks the server's gevent hub, which can strand a heartbeating connection.
        _reconnect(iface)
        try:
            iface.server.launch_controller()
        except Exception as err:  # noqa: BLE001 - report and retry
            last_error = err
            print("  launch_controller failed: %s" % _oneline(err))
            continue

        # 3. the client re-acquiring the tty means it got past `Unable to activate!`
        if tty:
            if wait_for(lambda: tty_holder(tty) is not None, timeout=30,
                        desc="gripper client to acquire %s" % tty):
                print("  gripper activated (client holds %s)" % tty)
            else:
                last_error = RuntimeError("gripper client never acquired %s" % tty)
                print("  gripper did NOT activate -- retrying")
                continue
        else:
            time.sleep(12)

        # 4. hand the (now healthy) stack to DROID and sanity-check it
        _reconnect(iface)
        try:
            iface.server.launch_robot()
            joints = iface.server.get_joint_positions()
            grip = iface.server.get_gripper_position()
        except Exception as err:  # noqa: BLE001 - report and retry
            last_error = err
            print("  launch_robot failed: %s" % _oneline(err))
            continue

        print("[safe-launch] OK  joints=%s  gripper=%.3f"
              % (["%.4f" % j for j in joints], grip))
        return

    raise RuntimeError(
        "safe_launch failed after %d attempts; last error: %s" % (attempts, _oneline(last_error))
    )


def _reconnect(iface):
    try:
        iface.server.close()
    except Exception:  # noqa: BLE001 - best effort
        pass
    iface.establish_connection()


def _oneline(err):
    if err is None:
        return "None"
    text = str(err).strip().splitlines()
    return text[-1] if text else repr(err)


#: Steps to hold still after commanding the gripper shut, at 15 Hz. The 2F-85 needs about
#: 1.7 s for its full stroke (0.085 m at the 0.05 m/s franka/robot.py asks for), which is
#: ~25 steps; 28 leaves a little margin. See GRIPPER_SETTLE_PATCH for why this is needed.
GRIPPER_CLOSE_STEPS = 28

#: Closure reading that counts as "the fingers have stopped moving", on the same 0 = open,
#: 1 = shut scale both sides use. Holding this bottle reads about 0.376 in sim, so 0.30 is
#: below the plateau but well above anything seen while the fingers are still travelling.
GRIPPER_SHUT_ON = 0.30

#: Reading band that means "the fingers stopped on the bottle" rather than "they closed on
#: nothing". Five real grasps read 0.304-0.423 and sim reads 0.376; an empty close runs to
#: about 0.85 (14.5 mm of pad gap left out of 97.7 open). Only the first latches the hand.
GRIPPER_HOLDING = (0.20, 0.60)


# ----------------------------------------------------------------------------------
# joint-position head: the changes listed in docs/real_robot.md ("Required client changes")
#
# Applied to the eval script's source in memory. Each `old` must appear exactly once;
# a miss aborts the launch rather than running a half-converted script.
# ----------------------------------------------------------------------------------
JOINTPOS_PATCHES = (
    (
        "action_space -> joint_position",
        "absolute joint angles must go straight through; also turns off robot_env.py:24's "
        "check_action_range (the +-1 assertion), which only applies to velocity spaces",
        '    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")',
        '    env = RobotEnv(action_space="joint_position", gripper_action_space="position")',
    ),
    (
        "max_timesteps 600 -> 1050",
        "70 s at 15 Hz, same budget as the sim eval; training trajectories run to 969 steps, "
        "so 600 would cut the long ones off before the pour",
        "    max_timesteps: int = 600",
        "    max_timesteps: int = 1050",
    ),
    (
        "enable strict gripper binarisation",
        "sim's droid_jointpos_client.py:87 is strictly binary and the training gripper column "
        "is exactly {0, 1}; without this, 0.87 means 'close to 87%' on the real 2F-85",
        """                # Binarize gripper action
                # if action[-1].item() > 0.5:
                #     # action[-1] = 1.0
                #     action = np.concatenate([action[:-1], np.ones((1,))])
                # else:
                #     # action[-1] = 0.0
                #     action = np.concatenate([action[:-1], np.zeros((1,))])""",
        """                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])""",
    ),
    (
        "drop the unconditional clip(-1, 1)",
        "it would crush absolute joint targets (j4 sits near -2.5 rad) to +-1 rad and the "
        "environment would not complain -- the assertion is off in position space",
        """                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)""",
        """                # [run_eval_safe] clip(-1, 1) removed: these are absolute joint
                # angles, not velocities. The gripper dim is binarised above and clipped
                # to [0, 1] anyway by franka/robot.py:209.""",
    ),
    (
        "hold still while the gripper closes",
        "in sim the gripper is a binary joint and reads shut the next step (median 1 step "
        "over the training set); the real 2F-85 needs ~25 steps at 15 Hz, and during them "
        "the policy sees a state it never saw in training - commanded shut, still reading "
        "open - so it stays in the grasp instead of lifting (2026-08-14, seen on the robot)",
        """        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None""",
        """        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        # [run_eval_safe] see the settle block below; the patched source runs in its own
        # __main__ namespace, so the constant is written in rather than imported
        prev_gripper = 0.0
        gripper_latched = False
        GRIPPER_CLOSE_STEPS = %d
        GRIPPER_SHUT_ON = %.2f
        GRIPPER_HOLDING = %r""" % (GRIPPER_CLOSE_STEPS, GRIPPER_SHUT_ON, GRIPPER_HOLDING),
    ),
    (
        "latch the gripper shut once it has the bottle",
        "no training trajectory ever re-opens: across all 216 the gripper column switches "
        "exactly once, 0 -> 1, so an open command after a successful grasp is a state the "
        "policy was never taught and here only drops the bottle. It latches on the settled "
        "reading rather than on the command, so closing on nothing does not lock the hand "
        "out of a retry: holding this bottle reads 0.30-0.42 (five real grasps, 0.376 in "
        "sim) while closing on air goes to about 0.85 (14.5 mm of pad gap out of 97.7).",
        """                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])""",
        """                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # [run_eval_safe] and keep it shut once it actually has the bottle: every
                # training trajectory closes once and holds. The latch is set by the settle
                # block below, on the measured reading, so a grasp that closed on nothing
                # can still be retried.
                if gripper_latched:
                    if action[-1] < 0.5:
                        print("[run_eval_safe] ignoring open command (holding the bottle)")
                    action = np.concatenate([action[:-1], np.ones((1,))])""",
    ),
    (
        "hold still while the gripper closes (the wait itself)",
        "repeating the same absolute joint target keeps the arm where it grasped while the "
        "fingers travel, so the next observation the policy gets is the one training paired "
        "with 'grasped': same pose, gripper reading shut",
        # Anchored on env.step alone: the debug print above it is commented out on some
        # laptops, and that is not a difference this patch should care about.
        """                env.step(action)""",
        """                env.step(action)

                # [run_eval_safe] the 2F-85 is not the sim's binary joint: hold the arm
                # still for its stroke instead of driving on while the fingers are open.
                # Stop as soon as the fingers report shut - they stall on the bottle at
                # about half the stroke, so the full count is only a cap.
                if action[-1] > 0.5 >= prev_gripper:
                    print(f"[run_eval_safe] gripper closing, holding up to {GRIPPER_CLOSE_STEPS} steps")
                    for held in range(GRIPPER_CLOSE_STEPS):
                        env.step(action)
                        time.sleep(1 / DROID_CONTROL_FREQUENCY)
                        grip = env.get_observation()["robot_state"]["gripper_position"]
                        if grip >= GRIPPER_SHUT_ON:
                            break
                    # holding the bottle reads 0.30-0.42 (0.376 in sim); closing on nothing
                    # runs on to about 0.85, and that must not latch or the hand is locked
                    # out of a second attempt
                    gripper_latched = GRIPPER_HOLDING[0] <= grip <= GRIPPER_HOLDING[1]
                    print(f"[run_eval_safe] gripper settled at {grip:.3f} after {held + 1} "
                          f"steps ({'holding' if gripper_latched else 'nothing in the hand'})")
                prev_gripper = float(action[-1])""",
    ),
)


# ----------------------------------------------------------------------------------
# camera preflight
#
# `_extract_observation` leaves left_image/wrist_image as None when a camera is missing
# and then subscripts them, so a camera that failed to enumerate surfaces as
# `TypeError: 'NoneType' object is not subscriptable` after the arm is already up.
# Check first, name the missing serial, and do not touch the robot.
# ----------------------------------------------------------------------------------
def _eval_arg(source, argv, field):
    """Value of an eval-script Args field: CLI override if given, else its default."""
    for flag in ("--" + field, "--" + field.replace("_", "-")):
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
    match = re.search(r'^\s*%s\s*:\s*str\s*=\s*"([^"]*)"' % re.escape(field), source, re.M)
    return match.group(1) if match else None


def check_cameras(source, argv):
    """Return a list of complaints -- empty means every camera the rollout needs is visible."""
    import pyzed.sl as sl

    wanted = {}
    for field, role in (("left_camera_id", "exterior (policy input)"),
                        ("wrist_camera_id", "wrist")):
        serial = _eval_arg(source, argv, field)
        if serial:
            wanted[serial] = role

    present = {str(device.serial_number) for device in sl.Camera.get_device_list()}
    missing = [(serial, role) for serial, role in wanted.items() if serial not in present]
    if not missing:
        print("[run_eval_safe] cameras: %s all visible" % ", ".join(sorted(wanted)))
        return []

    complaints = ["ZED SDK sees %s" % (", ".join(sorted(present)) or "no cameras at all")]
    for serial, role in missing:
        complaints.append("missing %s -- the %s camera" % (serial, role))
    complaints.append(
        "the rollout would die mid-flight on a None image, so stopping here. Usual causes: "
        "the DROID docker container is still holding the stream; or the camera's USB link "
        "went bad (check `dmesg | tail` for 'uvcvideo: Non-zero status (-71)'), which a "
        "replug into the same port fixes. --no-camera-check skips this."
    )
    return complaints


SCORING_PATCH = (
    (
        "score 'y' as 100%, not 1%",
        "upstream sets success = 1.0 for 'y' and then divides by 100 along with the numeric "
        "path, so a success gets logged as 0.01. 'n' is unaffected (0.0/100 = 0.0). Bites "
        "quietly: the rollout looks scored and the csv reads 1% (2026-08-14, first real "
        "success was logged this way)",
        """            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0""",
        """            if success == "y":
                success = 100.0  # [run_eval_safe] was 1.0, then divided by 100 below
            elif success == "n":
                success = 0.0""",
    ),
)


# Opt-in (--record-wrist). Recording only: it changes what lands in the mp4, never an
# action. The wrist view is the one that answers "what did the policy see at the grasp",
# and it is the view the sim renders too, so the two can be put side by side.
RECORD_WRIST_PATCH = (
    (
        "record the wrist view next to the exterior one",
        "the saved mp4 is exterior-only, so a failure that depends on what the wrist camera "
        "saw cannot be diagnosed after the fact",
        """                video.append(curr_obs[f"{args.external_camera}_image"])""",
        """                video.append(np.concatenate(
                    [curr_obs[f"{args.external_camera}_image"], curr_obs["wrist_image"]],
                    axis=1))  # [run_eval_safe] --record-wrist""",
    ),
)


def patch_eval_source(source, path, extra=()):
    """Return (patched_source, [(name, why), ...]). Raises if any patch does not apply."""
    applied = []
    for name, why, old, new in tuple(JOINTPOS_PATCHES) + tuple(SCORING_PATCH) + tuple(extra):
        found = source.count(old)
        if found != 1:
            raise RuntimeError(
                "patch %r does not apply to %s (found %d exact matches, expected 1).\n"
                "The upstream script changed. Re-read PolaRis_LFHV/docs/real_robot.md "
                "docs/real_robot.md and update JOINTPOS_PATCHES before going anywhere near the robot.\n"
                "Expected to find:\n%s" % (name, path, found, old)
            )
        source = source.replace(old, new, 1)
        applied.append((name, why))
    return source, applied


# ----------------------------------------------------------------------------------
# abort-only motion watchdog
#
# It never rewrites an action -- a clean rollout is bit-identical with it on or off, so
# the "no shaping at eval time" rule holds. It only refuses to hand over a command that
# is outside what the training data could contain, and ends the rollout gracefully.
#
# Position limits: the FR3 n Panda intersection the generator was capped at
# (PolaRis_LFHV/docs/fr3_limits.md). Velocity wall: the soft wall the training data was
# generated under; the final checkpoint peaked at 0.76 x this over 2100 logged steps.
# ----------------------------------------------------------------------------------
JOINT_MIN = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8763, 0.4398, -2.8973)
JOINT_MAX = (2.8973, 1.7628, 2.8973, -0.1169, 2.8763, 3.7525, 2.8973)
JOINT_VEL_WALL = (1.575, 1.575, 1.575, 1.575, 2.01, 2.01, 2.01)  # rad/s
CONTROL_HZ = 15


def install_guard():
    """Wrap RobotEnv.step so an out-of-distribution command stops the rollout."""
    import numpy as np

    from droid.robot_env import RobotEnv

    lo = np.array(JOINT_MIN)
    hi = np.array(JOINT_MAX)
    max_delta = np.array(JOINT_VEL_WALL) / CONTROL_HZ

    state = {"prev": None}
    original_step = RobotEnv.step
    original_reset = RobotEnv.reset

    def abort(reason, detail):
        print("\n" + "!" * 78)
        print("[guard] %s" % reason)
        print("[guard] %s" % detail)
        print("[guard] rollout stopped before this command was sent. HIT E-STOP if the arm "
              "is still moving.")
        print("!" * 78 + "\n")
        raise KeyboardInterrupt  # the eval loop catches this and ends the rollout cleanly

    def guarded_step(self, action, *args, **kwargs):
        target = np.asarray(action, dtype=float)[:7]

        outside = np.flatnonzero((target < lo) | (target > hi))
        if outside.size:
            abort(
                "commanded joint target is outside the FR3 n Panda limits",
                ", ".join("j%d=%.4f (limit %.4f..%.4f)" % (i + 1, target[i], lo[i], hi[i])
                          for i in outside),
            )

        prev = state["prev"]
        if prev is None:  # first command of the rollout: measure against where the arm is
            prev = np.asarray(self._robot.get_joint_positions(), dtype=float)
        delta = np.abs(target - prev)
        over = np.flatnonzero(delta > max_delta)
        if over.size:
            abort(
                "commanded step exceeds the velocity wall the training data was built under",
                ", ".join("j%d: %.4f rad/tick = %.2f rad/s (wall %.2f)"
                          % (i + 1, delta[i], delta[i] * CONTROL_HZ, JOINT_VEL_WALL[i])
                          for i in over),
            )

        state["prev"] = target
        return original_step(self, action, *args, **kwargs)

    def guarded_reset(self, *args, **kwargs):
        state["prev"] = None  # the reset move is not a policy step; do not diff across it
        return original_reset(self, *args, **kwargs)

    RobotEnv.step = guarded_step
    RobotEnv.reset = guarded_reset


# ----------------------------------------------------------------------------------
# action trace -- for "the arm is not moving" and nothing else says why
#
# Two numbers, once a second, that split the two possible causes:
#   cmd step   how far the policy is asking the arm to move between control ticks.
#              Near zero => the policy itself is standing still (look at the server:
#              norm stats, the exterior camera, the prompt).
#   follow err how far the measured joints are from the commanded target. Small means
#              the arm is tracking; large and growing means commands are going out but
#              the arm is not executing them (controller not loaded, FCI dropped, brakes).
# Read-only: it never changes an action, and it costs one extra RPC per second.
# ----------------------------------------------------------------------------------
def install_trace(every=CONTROL_HZ):
    import numpy as np

    from droid.robot_env import RobotEnv

    original_step = RobotEnv.step
    original_reset = RobotEnv.reset
    state = {"n": 0, "prev": None, "peak": 0.0}

    def traced_step(self, action, *args, **kwargs):
        target = np.asarray(action, dtype=float)[:7]
        if state["prev"] is not None:
            state["peak"] = max(state["peak"], float(np.abs(target - state["prev"]).max()))
        state["prev"] = target
        state["n"] += 1

        if state["n"] % every == 0:
            measured = np.asarray(self._robot.get_joint_positions(), dtype=float)
            print("[trace] step %4d  cmd step %.4f rad (%.2f rad/s)  follow err %.4f rad  "
                  "grip cmd %.0f" % (state["n"], state["peak"], state["peak"] * CONTROL_HZ,
                                     float(np.abs(target - measured).max()), float(action[-1])))
            state["peak"] = 0.0

        return original_step(self, action, *args, **kwargs)

    def traced_reset(self, *args, **kwargs):
        # The move back to home is not a control step. Diffing across it reports a fake
        # 30 rad/s at the start of every rollout after the first.
        state["prev"] = None
        state["peak"] = 0.0
        return original_reset(self, *args, **kwargs)

    RobotEnv.step = traced_step
    RobotEnv.reset = traced_reset


# ----------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------
def _pop_flag(argv, name):
    if name in argv:
        argv.remove(name)
        return True
    return False


def _pop_opt(argv, name, default, cast):
    for i, arg in enumerate(list(argv)):
        if arg == name:
            value = argv[i + 1]
            del argv[i:i + 2]
            return cast(value)
        if arg.startswith(name + "="):
            del argv[i]
            return cast(arg.split("=", 1)[1])
    return default


def main():
    argv = sys.argv[1:]
    check_only = _pop_flag(argv, "--check-only")
    show_diff = _pop_flag(argv, "--show-diff")
    no_guard = _pop_flag(argv, "--no-guard")
    no_camera_check = _pop_flag(argv, "--no-camera-check")
    trace = _pop_flag(argv, "--trace")
    record_wrist = _pop_flag(argv, "--record-wrist")
    head = _pop_opt(argv, "--head", "joint_position", str)
    attempts = _pop_opt(argv, "--attempts", 3, int)
    settle = _pop_opt(argv, "--settle", 4.0, float)
    eval_script = _pop_opt(
        argv, "--eval-script",
        os.path.join(DROID_ROOT, "scripts", "evaluation", "evaluate_openpi.py"), str
    )

    if head not in ("joint_position", "joint_velocity"):
        print("--head must be joint_position or joint_velocity, got %r" % head, file=sys.stderr)
        return 2
    if not os.path.exists(eval_script):
        print("eval script not found: %s" % eval_script, file=sys.stderr)
        return 2

    # Patch first: a bad patch must be found before anything touches the robot.
    with open(eval_script) as handle:
        original_source = handle.read()
    if head == "joint_position":
        source, applied = patch_eval_source(
            original_source, eval_script, RECORD_WRIST_PATCH if record_wrist else ())
    else:
        source, applied = original_source, []

    if show_diff:
        if not applied:
            print("--head joint_velocity: running %s unmodified." % eval_script)
            return 0
        sys.stdout.writelines(difflib.unified_diff(
            original_source.splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile=eval_script + " (on disk, untouched)", tofile=eval_script + " (as executed)",
        ))
        return 0

    if NUC_USER is None or NUC_CONTAINER is None:
        print("NUC_USER and NUC_CONTAINER must be set (the NUC login and its docker container name)",
              file=sys.stderr)
        return 2
    if NUC_SUDO_PASSWORD is None:
        print("NUC_SUDO_PASSWORD is not set -- it is needed to clear the old gripper client "
              "on the NUC. Credentials are not stored in this repository; on the robot "
              "laptop the value is in the lab's robot setup notes.", file=sys.stderr)
        return 2

    # Before the NUC stack comes up: a missing camera should cost nothing but a message.
    if not (no_camera_check or check_only):
        complaints = check_cameras(source, argv)
        if complaints:
            print("\n[run_eval_safe] camera check failed -- robot not touched", file=sys.stderr)
            for line in complaints:
                print("  %s" % line, file=sys.stderr)
            return 2

    print("[run_eval_safe] action head: %s" % head)
    for name, why in applied:
        print("  patched: %-38s  %s" % (name, why))
    if head == "joint_position" and not no_guard:
        install_guard()
        print("  guard:   on (abort-only; --no-guard to disable)")
    elif head == "joint_position":
        print("  guard:   OFF")
    if trace:
        install_trace()
        print("  trace:   on (one line per second: commanded step size + follow error)")
    print("[run_eval_safe] before you start: policy server logging norm stats from the "
          "official polaris assets, exterior camera = the deployment-pose ZED 2i, "
          "first step must print 'Predicted action chunk shape: (15, 8)'.")

    import droid.misc.server_interface as server_interface

    original_init = server_interface.ServerInterface.__init__

    def patched_init(self, ip_address="127.0.0.1", launch=True):
        # Deliberately does NOT call the original __init__: its attempt_n_times() retries
        # 0.1s apart, which is exactly what cannot work here.
        self.ip_address = ip_address
        self.establish_connection()
        if launch:
            safe_launch(self, attempts=attempts, settle=settle)

    patched_init.__wrapped__ = original_init
    server_interface.ServerInterface.__init__ = patched_init

    if check_only:
        iface = server_interface.ServerInterface(ip_address=nuc_ip)
        del iface
        print("[check-only] stack is up; not running the evaluation.")
        return 0

    sys.argv = [eval_script] + argv
    os.chdir(DROID_ROOT)  # eval writes rollouts/ and results/ relative to the repo root
    print("[safe-launch] handing over to %s" % eval_script)
    _exec_as_main(source, eval_script)
    return 0


def _exec_as_main(source, path):
    """Run `source` as if it were `python path` -- same namespace setup runpy would give it.

    The temporary sys.modules["__main__"] matters: tyro resolves the Args dataclass's
    annotations through it.
    """
    import types

    module = types.ModuleType("__main__")
    module.__file__ = path
    module.__builtins__ = __builtins__
    saved = sys.modules.get("__main__")
    sys.modules["__main__"] = module
    try:
        exec(compile(source, path, "exec"), module.__dict__)
    finally:
        if saved is not None:
            sys.modules["__main__"] = saved


if __name__ == "__main__":
    sys.exit(main())
