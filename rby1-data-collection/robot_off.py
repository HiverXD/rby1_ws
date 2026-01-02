import rby1_sdk
import numpy as np
import sys
import argparse
from rby1_sdk import *
from zero_pose import main as zero

def cb(rs):
    print(f"Timestamp: {rs.timestamp - rs.ft_sensor_right.time_since_last_update}")
    position = rs.position * 180 / 3.141592
    print(f"torso [deg]: {position[2:2 + 6]}")
    print(f"right arm [deg]: {position[8:8 + 7]}")
    print(f"left arm [deg]: {position[15:15 + 7]}")

def main(address, model_name, power, servo):
    print("Attempting to connect to the robot...")

    robot = rby1_sdk.create_robot(address, model_name)

    if not robot.connect():
        print("Error: Unable to establish connection to the robot at")
        sys.exit(1)

    print("Successfully connected to the robot")

    print("Starting state update...")
    robot.start_state_update(cb, 0.1)

    # robot.factory_reset_all_parameters()
    robot.set_parameter("default.acceleration_limit_scaling", "1.0")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "20")
    robot.set_parameter("default.angular_acceleration_limit", "10")
    robot.set_parameter("manipulability_threshold", "1e4")
    # robot.set_time_scale(1.0)

    print("parameters setting is done")

    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)

    # zero(address=address,
    #     model=model_name,
    #     power=power,
    #     servo=servo)

    robot.disable_control_manager()
    robot.power_off(power)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="09_demo_motion")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default='a', help="Robot Model Name (default: 'a')")
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default=".*",
        help="Servo name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    main(address=args.address, model_name = args.model, power=args.power, servo=args.servo)