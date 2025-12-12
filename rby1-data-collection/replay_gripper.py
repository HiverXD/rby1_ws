import argparse
import zmq
import logging
import rby1_sdk as rby
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from vr_control_state import VRControlState
import threading
from h5py_writer import H5Writer
import time
import h5py
from gripper import Gripper

@dataclass(frozen=True)
class Settings:
    dt: float = 0.1
    hand_offset: float = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    vr_control_local_port: int = 5005
    vr_control_meta_quest_port: int = 6000

    mobile_linear_acceleration_gain: float = 0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3


class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    vr_state: VRControlState = VRControlState()
    # H5 writer and recording stop-event stored here so other threads/handlers can access them
    h5_writer: Optional[H5Writer] = None
    rec_stop_event: Optional[threading.Event] = None


def connect_rby1(address: str, model: str = "a", no_head: bool = False):
    logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")
    robot = rby.create_robot(address, model)

    connected = robot.connect()
    if not connected:
        logging.critical("Failed to connect to RB-Y1. Exiting program.")
        exit(1)
    logging.info("Successfully connected to RB-Y1.")

    servo_pattern = "^(?!head_).*" if no_head else ".*"
    if not robot.is_power_on(servo_pattern):
        logging.warning("Robot power is off. Turning it on...")
        if not robot.power_on(servo_pattern):
            logging.critical("Failed to power on. Exiting program.")
            exit(1)
        logging.info("Power turned on successfully.")
    else:
        logging.info("Power is already on.")

    if not robot.is_servo_on(".*"):
        logging.warning("Servo is off. Turning it on...")
        if not robot.servo_on(".*"):
            logging.critical("Failed to turn on the servo. Exiting program.")
            exit(1)
        logging.info("Servo turned on successfully.")
    else:
        logging.info("Servo is already on.")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        logging.warning(f"Control Manager is in Fault state: {cm_state.name}. Attempting reset...")
        if not robot.reset_fault_control_manager():
            logging.critical("Failed to reset Control Manager. Exiting program.")
            exit(1)
        logging.info("Control Manager reset successfully.")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        logging.critical("Failed to enable Control Manager. Exiting program.")
        exit(1)
    logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")

    SystemContext.robot_model = robot.model()
    robot.start_state_update(robot_state_callback, 1 / Settings.dt)

    return robot


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.vr_state.joint_positions = robot_state.position # NOTE: where the current robot state is saved 
    SystemContext.vr_state.center_of_mass = robot_state.center_of_mass


def main(args: argparse.Namespace):
    MINIMUM_TIME = 3  # seconds
    frequency = 10  # Hz
    h5_file_path = '/media/nvidia/T7/Demo/demo_21.h5'
    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)

    # Initialize gripper unless explicitly disabled
    gripper = None
    if not getattr(args, 'no_gripper', False):
        try:
            for arm in ["left", "right"]:
                try:
                    robot.set_tool_flange_output_voltage(arm, 12)
                except Exception:
                    logging.debug(f"Failed to set 12V on flange {arm} (continuing)")
            time.sleep(0.5)
            gripper = Gripper()
            ok = gripper.initialize(verbose=True)
            if not ok:
                logging.warning("Gripper.initialize returned False — continuing without gripper.")
                gripper = None
            else:
                try:
                    gripper.homing()
                except Exception:
                    logging.exception("Gripper homing failed (continuing)")
                try:
                    gripper.start()
                except Exception:
                    logging.exception("Gripper start failed (continuing)")
        except Exception:
            logging.exception("Failed to initialize Gripper (continuing without)")
            gripper = None

    dry_run = getattr(args, 'dry_run', False)

    with h5py.File(h5_file_path, 'r') as f:
        gripper_dataset = f.get('samples/gripper_target', None)
        if gripper_dataset is None:
            gripper_dataset = f.get('samples/gripper_state', None)

        if gripper_dataset is None:
            logging.error("No gripper_target or gripper_state dataset found in HDF5")
            return

        n = len(gripper_dataset)
        logging.info(f"Replaying {n} gripper samples (dry_run={dry_run}, gripper_present={gripper is not None})")
        for i in range(n):
            start_time = time.time()
            try:
                g = np.asarray(gripper_dataset[i])
                if g.ndim > 1:
                    g = g.ravel()
                # g[0] = right, g[1] = left
                logging.info(f"Sample {i}: gripper_target={g}")
                # allow optional inversion of open/close mapping
                applied = g
                if getattr(args, 'invert_gripper', False):
                    try:
                        applied = 1.0 - np.array(g)
                    except Exception:
                        logging.exception("Failed to invert gripper target for sample %d", i)
                    logging.info(f"Sample {i}: inverted gripper_target={applied}")

                if not dry_run and gripper is not None:
                    try:
                        gripper.set_normalized_target(applied)
                    except Exception:
                        logging.exception("Failed to set gripper target for sample %d", i)
                else:
                    if dry_run:
                        logging.info("Dry-run: not sending to gripper")
                    else:
                        logging.info("No gripper initialized; skipping set_normalized_target")
            except Exception:
                logging.exception("Failed to read gripper sample %d", i)

            time.sleep(max(0, 1/frequency - (time.time() - start_time)))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RB-Y1 VR Control Launcher")
    parser.add_argument(
        "--rby1", default="192.168.30.1:50051", type=str,
        help="gRPC address of the RB-Y1 robot (default: 192.168.30.1:50051)"
    )
    parser.add_argument(
        "--rby1_model", default="a", type=str,
        help="Model type of the RB-Y1 robot (default: a)"
    )
    parser.add_argument(
        "--no_head", action="store_true", 
        help="Run without controlling the head"
    )
    parser.add_argument(
        "--no_gripper", action="store_true",
        help="Do not initialize or control the local gripper"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Do not send commands to gripper; just log targets"
    )
    parser.add_argument(
        "--invert_gripper", action="store_true",
        help="Invert gripper open/close values (apply 1 - value before sending)"
    )

    args = parser.parse_args()

    main(args)