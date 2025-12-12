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
    MINIMUM_TIME = 0.5  # seconds
    frequency = 1  # Hz
    h5_file_path = '/media/nvidia/T7/Demo/demo_17.h5'

    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)

    with h5py.File(h5_file_path, 'r') as f:
        robot_dataset = f.get('samples/robot_target_joints', None)
        base_dataset = f.get('samples/base_state', None)

        if robot_dataset is not None:
            n = len(robot_dataset)
            for i in range(n):
                start_time = time.time()
                data = robot_dataset[i]
                torso_joints = data[2:8]
                right_joints = data[8:15]
                left_joints = data[15:22]

                # mobility command from base_state (linear_x, linear_y, angular_z)
                mobility_builder = None
                if base_dataset is not None:
                    try:
                        b = np.asarray(base_dataset[i])
                        # expected b shape (3,) -> [lx, ly, az]
                        lin = np.array([b[0], b[1]])
                        ang = float(b[2])
                        # main.py negates mobile velocities when sending; reproduce that
                        mobility_builder = rby.SE2VelocityCommandBuilder()
                        mobility_builder.set_velocity(-lin, -ang).set_minimum_time(Settings.dt * 1.01)
                    except Exception:
                        logging.exception("Failed to read/apply base_state for sample %d", i)

                body_builder = (
                    rby.BodyComponentBasedCommandBuilder()
                    .set_torso_command(
                        rby.JointPositionCommandBuilder()
                        .set_minimum_time(MINIMUM_TIME)
                        .set_position(torso_joints)
                    )
                    .set_right_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_minimum_time(MINIMUM_TIME)
                        .set_position(right_joints)
                    )
                    .set_left_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_minimum_time(MINIMUM_TIME)
                        .set_position(left_joints)
                    )
                )

                cbb = rby.ComponentBasedCommandBuilder()
                if mobility_builder is not None:
                    cbb.set_mobility_command(mobility_builder)
                cbb.set_body_command(body_builder)

                rc = rby.RobotCommandBuilder().set_command(cbb)

                try:
                    rv = robot.send_command(rc, 10).get()
                except Exception:
                    logging.exception("Failed to send robot command for sample %d", i)

                time.sleep(max(0, 1/frequency - (time.time() - start_time)))

        else:
            # No robot joint dataset; if base dataset exists, play base-only
            if base_dataset is not None:
                for i in range(len(base_dataset)):
                    start_time = time.time()
                    try:
                        b = np.asarray(base_dataset[i])
                        lin = np.array([b[0], b[1]])
                        ang = float(b[2])
                        mobility = rby.SE2VelocityCommandBuilder()
                        mobility.set_velocity(-lin, -ang).set_minimum_time(Settings.dt * 1.01)
                        rc = rby.RobotCommandBuilder().set_command(
                            rby.ComponentBasedCommandBuilder().set_mobility_command(mobility)
                        )
                        robot.send_command(rc, 10).get()
                    except Exception:
                        logging.exception("Failed to send base-only command for sample %d", i)
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

    args = parser.parse_args()

    main(args)