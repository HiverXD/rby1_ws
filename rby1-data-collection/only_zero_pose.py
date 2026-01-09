import rby1_sdk as rby
from helper import *
import argparse
import numpy as np
import logging
from setup import Settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 바로 zero pose로 이동
def main(address, model, power, servo):
    robot = initialize_robot(address, model, power, servo)

    model = robot.model()
    torso_dof = len(model.torso_idx)
    right_arm_dof = len(model.right_arm_idx)
    left_arm_dof = len(model.left_arm_idx)
    head_dof = len(model.head_idx)

    movej(
        robot,
        np.zeros(torso_dof),
        np.zeros(right_arm_dof),
        np.zeros(left_arm_dof),
        np.zeros(head_dof),
        minimum_time=10,
    )

    # q_joint_waist = np.array([0, 0, 0, 0, 0, 0]) 
    # q_joint_right_arm = np.array([0, 0, 0, 0, 0, 0, 0])
    # q_joint_left_arm = np.array([0, 0, 0, 0, 0, 0, 0])

    # # q_joint_right_arm = np.array([-13.2, -35.7, 13.3, -138.5, -59.2, 101.4, -72.2]) * D2R
    # # q_joint_left_arm = np.array([-13.2, 35.7, -13.3, -138.5, 59.2, 101.4, 72.2]) * D2R
    # # Combine joint positions
    # q_init = np.concatenate([q_joint_waist, q_joint_right_arm, q_joint_left_arm])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="07_impedance_control")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
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

    main(
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo
    )
