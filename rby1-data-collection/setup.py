import numpy as np
from dataclasses import dataclass
from typing import Union, Optional
import threading
import rby1_sdk as rby
from h5py_writer import H5Writer
from vr_control_state import VRControlState
from lerobot_handler import LeRobotDataHandler

@dataclass(frozen=True)
class Settings:
    dt: float = 0.1
    hand_offset: float = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    vr_control_local_port: int = 5005
    vr_control_meta_quest_port: int = 6000

    mobile_linear_acceleration_gain: float = -0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3

    rec_fps: int = 10

    MAX_POINT: int = None

    # Initial poses in degrees
    torso_init_pose: tuple = (0.0, 20.0, -40.0, 20.0, 0.0, 0.0)
    right_arm_init_pose: tuple = (0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    left_arm_init_pose: tuple = (0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    torso_head_init_pose: tuple = (0.0, 0.0)
    bimanual_head_init_pose: tuple = (0.0, 40.0)
    
    shoulder_pitch_angle: float = 70.0
    shoulder_roll_angle: float = 30.0
    elbow_angle: float = -100.0
    wrist_angle: float = -70.0

    right_arm_midpoint1 = np.deg2rad([shoulder_pitch_angle, -shoulder_roll_angle, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])
    left_arm_midpoint1 = np.deg2rad([shoulder_pitch_angle, shoulder_roll_angle, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])

    right_arm_midpoint2 = np.deg2rad([0.0, -15.0, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])
    left_arm_midpoint2 = np.deg2rad([0.0, 15.0, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])

    body_init_pose: float = np.deg2rad(torso_init_pose + right_arm_init_pose + left_arm_init_pose)

class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    vr_state: VRControlState = VRControlState()
    # H5 writer and recording stop-event stored here so other threads/handlers can access them
    h5_writer: Optional[H5Writer] = None
    #lerobot
    lerobot_handler: Optional[LeRobotDataHandler] = None
    rec_stop_event: Optional[threading.Event] = None