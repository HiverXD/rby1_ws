#!/usr/bin/env python3
"""
Unified replay script for RBY1 demonstrations.

Replays H5-recorded demos back to the robot.  Supports four replay modes
that can be selected via ``--mode``:

  joints   – replay robot_target_joints (torso + arms)  [default]
  base     – replay base_state (mobility) + robot_target_joints
  gripper  – replay gripper_target only
  all      – replay joints + base + gripper together

Usage:
  python replay.py --rby1 192.168.0.50:50051 --h5 /path/to/demo.h5 --mode joints
  python replay.py --rby1 192.168.0.50:50051 --h5 /path/to/demo.h5 --mode all
"""

import argparse
import logging
import time

import h5py
import numpy as np
import rby1_sdk as rby

from helper import initialize_robot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)


# ---------------------------------------------------------------------------
# Gripper helper (best-effort)
# ---------------------------------------------------------------------------

def _init_gripper(robot, no_gripper: bool = False):
    """Try to initialise the local Dynamixel gripper.  Returns None on failure."""
    if no_gripper:
        return None
    try:
        from gripper import Gripper

        for arm in ("left", "right"):
            try:
                robot.set_tool_flange_output_voltage(arm, 12)
            except Exception:
                pass
        time.sleep(0.5)

        gr = Gripper()
        if not gr.initialize(verbose=True):
            logging.warning("Gripper.initialize() returned False — continuing without gripper.")
            return None
        try:
            gr.homing()
        except Exception:
            pass
        try:
            gr.start()
        except Exception:
            pass
        return gr
    except Exception:
        logging.warning("Could not initialise Gripper (continuing without).")
        return None


# ---------------------------------------------------------------------------
# Per-mode replay implementations
# ---------------------------------------------------------------------------

def replay_joints(robot, h5_path: str, *, frequency: float = 1.0, minimum_time: float = 0.5):
    """Replay robot_target_joints (torso + both arms).  Gripper via local driver."""
    gripper = _init_gripper(robot, no_gripper=False)

    with h5py.File(h5_path, "r") as f:
        dataset = f["samples/robot_target_joints"]
        n = len(dataset)
        logging.info(f"[joints] Replaying {n} samples from {h5_path}")

        for i in range(n):
            t0 = time.time()
            data = dataset[i]
            torso = data[2:8]
            right_arm = data[8:15]
            left_arm = data[15:22]
            gripper_state = np.asarray(data[0:2], dtype=float)

            rc = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(torso))
                    .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(right_arm))
                    .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(left_arm))
                )
            )
            try:
                robot.send_command(rc, 10).get()
            except Exception:
                logging.exception("Failed to send joint command at sample %d", i)

            # Gripper (best-effort)
            if gripper is not None:
                try:
                    gripper.set_normalized_target(gripper_state)
                except Exception:
                    pass

            time.sleep(max(0, 1 / frequency - (time.time() - t0)))


def replay_base(robot, h5_path: str, *, frequency: float = 1.0, minimum_time: float = 0.5, dt: float = 0.1):
    """Replay robot_target_joints + base_state (mobility)."""
    with h5py.File(h5_path, "r") as f:
        joint_ds = f.get("samples/robot_target_joints")
        base_ds = f.get("samples/base_state")

        if joint_ds is not None:
            n = len(joint_ds)
            logging.info(f"[base] Replaying {n} samples (joints + mobility)")

            for i in range(n):
                t0 = time.time()
                data = joint_ds[i]
                torso = data[2:8]
                right_arm = data[8:15]
                left_arm = data[15:22]

                cbb = rby.ComponentBasedCommandBuilder()

                # Mobility
                if base_ds is not None:
                    try:
                        b = np.asarray(base_ds[i])
                        lin = np.array([b[0], b[1]])
                        ang = float(b[2])
                        mob = rby.SE2VelocityCommandBuilder().set_velocity(-lin, -ang).set_minimum_time(dt * 1.01)
                        cbb.set_mobility_command(mob)
                    except Exception:
                        logging.exception("Failed to read base_state at sample %d", i)

                cbb.set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(torso))
                    .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(right_arm))
                    .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(left_arm))
                )

                try:
                    robot.send_command(rby.RobotCommandBuilder().set_command(cbb), 10).get()
                except Exception:
                    logging.exception("Failed to send command at sample %d", i)

                time.sleep(max(0, 1 / frequency - (time.time() - t0)))

        elif base_ds is not None:
            n = len(base_ds)
            logging.info(f"[base] No joint data — replaying {n} base-only samples")
            for i in range(n):
                t0 = time.time()
                try:
                    b = np.asarray(base_ds[i])
                    mob = rby.SE2VelocityCommandBuilder().set_velocity(
                        -np.array([b[0], b[1]]), -float(b[2])
                    ).set_minimum_time(dt * 1.01)
                    rc = rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_mobility_command(mob)
                    )
                    robot.send_command(rc, 10).get()
                except Exception:
                    logging.exception("Failed at base sample %d", i)
                time.sleep(max(0, 1 / frequency - (time.time() - t0)))
        else:
            logging.error("H5 has neither robot_target_joints nor base_state")


def replay_gripper(robot, h5_path: str, *, frequency: float = 10.0,
                   no_gripper: bool = False, dry_run: bool = False, invert: bool = False):
    """Replay gripper_target only."""
    gripper = _init_gripper(robot, no_gripper=no_gripper)

    with h5py.File(h5_path, "r") as f:
        ds = f.get("samples/gripper_target") or f.get("samples/gripper_state")
        if ds is None:
            logging.error("No gripper_target or gripper_state found in HDF5")
            return

        n = len(ds)
        logging.info(f"[gripper] Replaying {n} samples (dry_run={dry_run})")
        for i in range(n):
            t0 = time.time()
            g = np.asarray(ds[i]).ravel()
            if invert:
                g = 1.0 - g
            logging.info(f"Sample {i}: gripper={g}")
            if not dry_run and gripper is not None:
                try:
                    gripper.set_normalized_target(g)
                except Exception:
                    logging.exception("Failed at gripper sample %d", i)
            time.sleep(max(0, 1 / frequency - (time.time() - t0)))


def replay_all(robot, h5_path: str, *, frequency: float = 1.0, minimum_time: float = 0.5, dt: float = 0.1):
    """Replay joints + base + gripper simultaneously."""
    gripper = _init_gripper(robot, no_gripper=False)

    with h5py.File(h5_path, "r") as f:
        joint_ds = f.get("samples/robot_target_joints")
        base_ds = f.get("samples/base_state")
        grip_ds = f.get("samples/gripper_target") or f.get("samples/gripper_state")

        if joint_ds is None:
            logging.error("samples/robot_target_joints not found in H5")
            return

        n = len(joint_ds)
        logging.info(f"[all] Replaying {n} samples (joints + base + gripper)")

        for i in range(n):
            t0 = time.time()
            data = joint_ds[i]
            torso = data[2:8]
            right_arm = data[8:15]
            left_arm = data[15:22]

            cbb = rby.ComponentBasedCommandBuilder()

            # Mobility
            if base_ds is not None and i < len(base_ds):
                try:
                    b = np.asarray(base_ds[i])
                    mob = rby.SE2VelocityCommandBuilder().set_velocity(
                        -np.array([b[0], b[1]]), -float(b[2])
                    ).set_minimum_time(dt * 1.01)
                    cbb.set_mobility_command(mob)
                except Exception:
                    pass

            cbb.set_body_command(
                rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(torso))
                .set_right_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(right_arm))
                .set_left_arm_command(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(left_arm))
            )

            try:
                robot.send_command(rby.RobotCommandBuilder().set_command(cbb), 10).get()
            except Exception:
                logging.exception("Failed at sample %d", i)

            # Gripper
            if gripper is not None and grip_ds is not None and i < len(grip_ds):
                try:
                    gripper.set_normalized_target(np.asarray(grip_ds[i]).ravel())
                except Exception:
                    pass

            time.sleep(max(0, 1 / frequency - (time.time() - t0)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RBY1 unified demo replay")
    parser.add_argument("--rby1", type=str, default="192.168.0.50:50051", help="Robot gRPC address")
    parser.add_argument("--rby1_model", type=str, default="a", help="Robot model (default: a)")
    parser.add_argument("--no_head", action="store_true", help="Exclude head servos")
    parser.add_argument("--h5", type=str, required=True, help="Path to H5 demo file")
    parser.add_argument("--mode", choices=["joints", "base", "gripper", "all"], default="joints",
                        help="Replay mode (default: joints)")
    parser.add_argument("--frequency", type=float, default=1.0, help="Replay Hz (default: 1.0)")
    parser.add_argument("--no_gripper", action="store_true", help="Skip gripper init")
    parser.add_argument("--dry_run", action="store_true", help="(gripper mode) log only, don't actuate")
    parser.add_argument("--invert_gripper", action="store_true", help="(gripper mode) invert open/close")

    args = parser.parse_args()

    robot = initialize_robot(args.rby1, args.rby1_model)

    if args.mode == "joints":
        replay_joints(robot, args.h5, frequency=args.frequency)
    elif args.mode == "base":
        replay_base(robot, args.h5, frequency=args.frequency)
    elif args.mode == "gripper":
        replay_gripper(robot, args.h5, frequency=args.frequency,
                       no_gripper=args.no_gripper, dry_run=args.dry_run, invert=args.invert_gripper)
    elif args.mode == "all":
        replay_all(robot, args.h5, frequency=args.frequency)


if __name__ == "__main__":
    main()
