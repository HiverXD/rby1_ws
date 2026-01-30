import numpy as np 
from sensor_msgs.msg import Image
import os
import yaml
import logging
import rby1_sdk as rby
from typing import Union

def rosimg_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image → np.ndarray (HxWxC, uint8)."""
    dtype = np.uint8  # adjust if msg.encoding differs
    img = np.frombuffer(msg.data, dtype=dtype)
    img = img.reshape((msg.height, msg.width, -1))  # works for RGB/BGR/mono # (H, W, C)
    return img


def get_next_h5_path(base_dir="/home/nvidia/rby1_ws/rby1-data-collection/data"):
# def get_next_h5_path(base_dir="/media/nvidia/T7/Demo"):
    # Count existing .h5 files
    existing = [f for f in os.listdir(base_dir) if f.endswith(".h5")]
    next_index = len(existing)
    # Create new path
    return os.path.join(base_dir, f"demo_{next_index}.h5")


def elbows_bending_check(robot: rby.Robot_A) -> bool:
    """
    Checks if elbow angles exceed a threshold to determine if a movement should be skipped.

    This function encapsulates the entire logic:
    1. Reads the current joint angles from the robot state.
    2. Gets robot model details (DOFs).
    3. Reads the angle threshold from 'config.yaml'.
    4. Compares the current elbow angles against the threshold.

    Args:
        robot: The robot object from which to get the state.

    Returns:
        True if the movement should be skipped (angle exceeded), False otherwise.
    """
    if robot is None:
        return False
        
    robot_state = robot.get_state()
    if robot_state is None:
        return False

    joint_angles = robot_state.position
    if joint_angles is None or joint_angles.size == 0:
        # If joint data is unavailable, assume no skip for now.
        # Depending on safety requirements, this might need to raise an error or return True to force a skip.
        return False

    try:
        # Get DOFs from robot model
        model = robot.model()
        head_dof = len(model.head_idx)
        torso_dof = len(model.torso_idx)
        right_arm_dof = len(model.right_arm_idx)

        # Extract elbow angles using hardcoded offsets.
        # This assumes a fixed joint order. A more robust solution would dynamically
        # look up joint indices by name if the SDK provides such functionality.
        right_elbow_angle = joint_angles[head_dof + torso_dof + 3]
        left_elbow_angle = joint_angles[head_dof + torso_dof + right_arm_dof + 3]

        # Load threshold from YAML
        try:
            with open('rby1-data-collection/config.yaml', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            elbow_threshold_deg = config.get('elbow_angle_threshold_deg', 90)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load or parse config.yaml: {e}. Using default threshold 90 deg.")
            elbow_threshold_deg = 90

        elbow_threshold_rad = np.deg2rad(elbow_threshold_deg)

        # Perform the check
        if abs(right_elbow_angle) > elbow_threshold_rad or abs(left_elbow_angle) > elbow_threshold_rad:
            return True

    except IndexError:
        logging.warning("Joint angle index out of bounds when checking elbow angles. Check joint_angles size and DOF values.")
    except Exception as e:
        logging.warning(f"An unexpected error occurred in check_elbows_for_skip: {e}")

    return False

