import argparse
import logging
import zmq
import time
import threading
from dataclasses import dataclass
import rby1_sdk as rby
import socket
from typing import Union, Optional
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from gripper import Gripper
from vr_control_state import VRControlState
import pickle
# from lerobot_handler import LeRobotDataHandler
from rclpy.executors import SingleThreadedExecutor  # or MultiThreadedExecutor

# demo writer 
from h5py_writer import H5Writer

# ROS2 Camera subscriber
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import time

from utils import *

from rclpy.qos import qos_profile_sensor_data
import threading
import subprocess
import os
import signal
import shlex
import re
import copy
from helper import * 

class HeadCamSub(Node):
    def __init__(self):
        super().__init__('head_cam_sub')
        self._lock = threading.Lock()
        self._got_first_color = threading.Event()
        self._got_first_depth = threading.Event()
        self._seq = 0
        
        # Subscribe to color image
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_cb,
            qos_profile_sensor_data,
        )
        
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_cb,
            qos_profile_sensor_data,
        )
        
        self.curr_frame = None
        self.curr_depth = None
        self.last_stamp = None  # (sec, nsec) from ROS header
        self.last_depth_stamp = None

    def color_cb(self, msg: Image):
        arr = rosimg_to_numpy(msg)  # should be HxWx3 uint8
        #with self._lock:
        self.curr_frame = arr  # store latest
        self.last_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._seq += 1
        self._got_first_color.set()
        # try:
        #     # log arrival of a new color frame (seq and composed timestamp)
        #     ts = float(self.last_stamp[0]) + float(self.last_stamp[1]) * 1e-9
        #     logging.info(f"[HeadCamSub] color frame arrived seq={self._seq} ts={ts:.6f}")
        # except Exception:
        #     pass

    def depth_cb(self, msg: Image):
        # Depth image is typically uint16 Z16 format
        depth_arr = np.frombuffer(msg.data, dtype=np.uint16)
        depth_arr = depth_arr.reshape((msg.height, msg.width))
        #with self._lock:
        self.curr_depth = depth_arr
        self.last_depth_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._got_first_depth.set()
        # try:
        #     ts = float(self.last_depth_stamp[0]) + float(self.last_depth_stamp[1]) * 1e-9
        #     logging.info(f"[HeadCamSub] depth frame arrived ts={ts:.6f}")
        # except Exception:
        #     pass

    # helper to safely fetch a copy
    def get_frame_copy(self):
        #with self._lock:
        if self.curr_frame is None:
            return None, None, None
        return copy.deepcopy(self.curr_frame), self.last_stamp, self._seq
    
    def get_depth_copy(self):
        #with self._lock:
        if self.curr_depth is None:
            return None, None
        return copy.deepcopy(self.curr_depth), self.last_depth_stamp


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


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

    rec_fps: int = 10

    MAX_POINT: int = None

    # Initial poses in degrees
    torso_init_pose: tuple = (0.0, 20.0, -40.0, 20.0, 0.0, 0.0)
    right_arm_init_pose: tuple = (0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    left_arm_init_pose: tuple = (0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    torso_head_init_pose: tuple = (0.0, 0.0)
    bimanual_head_init_pose: tuple = (0.0, 40.0)
    
    shoulder_pitch_angle = 70.0
    shoulder_roll_angle = 30.0
    elbow_angle = -100.0
    wrist_angle = -70.0

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
    #lerobot_handler: Optional[LeRobotDataHandler] = None
    rec_stop_event: Optional[threading.Event] = None


def open_zmq_pub_socket(bind_address: str) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)
    logging.info(f"ZMQ PUB server opened at {bind_address}")
    return socket


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.vr_state.joint_positions = robot_state.position # NOTE: where the current robot state is saved 
    SystemContext.vr_state.center_of_mass = robot_state.center_of_mass


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


def setup_meta_quest_udp_communication(local_ip: str, local_port: int, meta_quest_ip: str, meta_quest_port: int,
                                       power_off=None):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        target_info = {
            "ip": local_ip,
            "port": local_port
        }
        message = json.dumps(target_info).encode('utf-8')
        sock.sendto(message, (meta_quest_ip, meta_quest_port))
        logging.info(f"Sent local PC info to Meta Quest: {target_info}")

    def udp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.bind((local_ip, local_port))
            logging.info(f"UDP server running to receive Meta Quest Controller data... {local_ip}:{local_port}")
            while True:
                data, addr = server_sock.recvfrom(4096)
                udp_msg = data.decode('utf-8')
                try:
                    SystemContext.vr_state.controller_state = json.loads(udp_msg)
                    if "left" in SystemContext.vr_state.controller_state["hands"]:
                        buttons = SystemContext.vr_state.controller_state["hands"]["left"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_left_a_pressed |= primary_button
                        SystemContext.vr_state.event_left_b_pressed |= secondary_button

                        # HACK: record/stop trigger
                        if primary_button: # NOTE: equivalent to 'X' in left controller
                            if power_off is not None:
                                logging.warning("Left X button pressed. Shutting down power.")
                                power_off()
                            pass

                    if "right" in SystemContext.vr_state.controller_state["hands"]:
                        buttons = SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_right_a_pressed |= primary_button
                        SystemContext.vr_state.event_right_b_pressed |= secondary_button

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to decode JSON: {e} (from {addr}) - received data: {message[:100]}")

    thread = threading.Thread(target=udp_server, daemon=True)
    thread.start()


def handle_vr_button_event(robot: Union[rby.Robot_A, rby.Robot_M], no_head: bool):
    global torso_mode
    model = robot.model()
    torso_dof = len(model.torso_idx)
    head_dof = len(model.head_idx)

    if SystemContext.vr_state.event_right_a_pressed:
        logging.info("Right A button pressed. Torso Mode initialized. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):
            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(Settings.body_init_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(2)
                )
            )
            # If head is present, move head to zero pose on initialization
            if (not no_head) and getattr(SystemContext, "robot_model", None) is not None:
                try:
                    head_len = len(SystemContext.robot_model.head_idx)
                    if head_len > 0:
                        cbc.set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_position(np.deg2rad(Settings.torso_head_init_pose))
                            .set_minimum_time(2)
                        )
                except Exception as e:
                    logging.warning(f"Failed to set head zero pose on init: {e}")

            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        torso_mode = True
        SystemContext.vr_state.is_initialized = True
        SystemContext.vr_state.is_stopped = False

    elif SystemContext.vr_state.event_left_b_pressed:
        logging.info("Left Y button pressed. Bimanual Mode initialized. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):
            movej(
            robot,
            np.zeros(torso_dof),
            Settings.right_arm_midpoint1,
            Settings.left_arm_midpoint1,
            np.zeros(head_dof),
            minimum_time=10,
    )

            movej(
                robot,
                np.zeros(torso_dof),
                Settings.right_arm_midpoint2,
                Settings.left_arm_midpoint2,
                np.zeros(head_dof),
                minimum_time=10,
            )
            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(Settings.body_init_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(2)
                )
            )

            # If head is present, move head to zero pose on initialization
            if (not no_head) and getattr(SystemContext, "robot_model", None) is not None:
                try:
                    head_len = len(SystemContext.robot_model.head_idx)
                    if head_len > 0:
                        cbc.set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_position(np.deg2rad(Settings.bimanual_head_init_pose))
                            .set_minimum_time(2)
                        )
                except Exception as e:
                    logging.warning(f"Failed to set head zero pose on init: {e}")

            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        torso_mode = False
        SystemContext.vr_state.is_initialized = True
        SystemContext.vr_state.is_stopped = False

    elif SystemContext.vr_state.event_right_b_pressed:
        logging.info("Right B button pressed. Stopping and saving recording.")
        # Stop the demo logger (signal its stop event) and flush/close the H5 file
        try:
            if SystemContext.rec_stop_event is not None:
                SystemContext.rec_stop_event.set()
                logging.info("Signaled demo logger to stop")
        except Exception as e:
            logging.warning(f"Failed to signal demo logger: {e}")

        try:
            if SystemContext.h5_writer is not None:
                SystemContext.h5_writer.stop()
                logging.info("H5 writer stopped and file saved")
        except Exception as e:
            logging.warning(f"Failed to stop H5 writer: {e}")
        # lerobot
        '''try:
            if SystemContext.lerobot_handler is not None:
                SystemContext.lerobot_handler.save_episode()
                logging.info("LeRobot handler saved episode")
        except Exception as e:
            logging.warning(f"Failed to save LeRobot episode: {e}")'''
        

        SystemContext.vr_state.is_stopped = True

    else:
        return False

    SystemContext.vr_state.event_right_a_pressed = False
    SystemContext.vr_state.event_right_b_pressed = False
    SystemContext.vr_state.event_left_a_pressed = False
    SystemContext.vr_state.event_left_b_pressed = False

    return True


def pose_to_se3(position, rotation_quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = position
    return T


def average_so3_slerp(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    # 두 회전을 Rotation 객체로 변환
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)

    # 보간 설정: t=0 => rot1, t=1 => rot2
    slerp = Slerp([0, 1], R.concatenate([rot1, rot2]))

    # 평균값은 중간지점 t=0.5
    rot_avg = slerp(0.5)
    return rot_avg.as_matrix()


def publish_gv(sock: zmq.Socket):
    while True:
        sock.send(pickle.dumps(SystemContext.vr_state))
        time.sleep(0.1)

# NOTE
#lerobot_handler 나중에 추가
def start_demo_logger(gripper: Gripper | None, h5_writer, robot, fps: int = 30) -> threading.Event:
    """
    Starts a daemon thread that prints a dict with:
      - robot_state.position
      - gripper.get_state()
    at ~fps Hz (default 30). Returns a stop_event you can set() to stop it.
    """
    print("start demo logger on\n")
    period = 1.0 / float(fps)
    stop_event = threading.Event()

    # Get robot dynamics for IK computation
    dyn_robot = robot.get_dynamics()
    robot_model = robot.model()

    # Import PCD utilities
    from pcd_utils import rgbd_to_pointcloud, REALSENSE_D435_INTRINSICS, REALSENSE_D435_INTRINSICS_848x480


    # FIXME: create sub node to retrieve image 
    headcam_sub = HeadCamSub()  # subclass of rclpy.node.Node
    executor = SingleThreadedExecutor()
    executor.add_node(headcam_sub)

    spin_thread = threading.Thread(target=executor.spin, name="ros2_spin", daemon=True)
    spin_thread.start()


    def _loop():
        next_t = time.perf_counter()

        robot_pos_prev = None

        while not stop_event.is_set():
            # Thread-safe snapshot of latest camera frames
            frame, frame_stamp, frame_seq = headcam_sub.get_frame_copy()
            depth, depth_stamp = headcam_sub.get_depth_copy()
            # Robot joint positions - 현재 포지션 읽기

            robot_pos = None
            try:
                if SystemContext.vr_state.joint_positions is not None:
                    # robot_pos = np.array(SystemContext.robot_state.position, dtype=float)
                # elif SystemContext.vr_state.joint_positions.size > 0:
                    # fallback to the latest joints cached in vr_state
                    robot_pos = np.array(SystemContext.vr_state.joint_positions, dtype=float)
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read robot position: {e}")
            

            # 이전 스텝의 robot_target_joints를 현재 robot_pos로 업데이트
            h5_writer.update_previous_target(robot_pos)

            robot_target_joints = robot_pos

            # Gripper encoders (actual measured state)
            grip = None
            try:
                if gripper is not None and hasattr(gripper, "get_state"):
                    grip = gripper.get_state()  # expected to return np.ndarray or None
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read gripper state: {e}")

            # base velocity
            base_state = None
            try:
                if SystemContext.vr_state.mobile_linear_velocity is not None:
                    linear_vel = np.array(SystemContext.vr_state.mobile_linear_velocity, dtype=float)
                    linear_vel = np.squeeze(linear_vel)
                    angular_vel = np.array(SystemContext.vr_state.mobile_angular_velocity, dtype=float)
                    base_state = np.concatenate((linear_vel, [angular_vel]), axis=0)
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read base velocity: {e}")

            # Initialize button states
            right_arm_pressed = False
            right_grip_pressed = False
            left_arm_pressed = False
            left_grip_pressed = False

            # 왼손 오른손 작동 그립 넷 중 하나 눌렀으면 데이터 수집
            if "hands" in SystemContext.vr_state.controller_state and "right" in SystemContext.vr_state.controller_state["hands"]:    
                right_arm_pressed = SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]["grip"] > 0.8
                right_grip_pressed = SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]["trigger"] > 0.8

            if "hands" in SystemContext.vr_state.controller_state and "left" in SystemContext.vr_state.controller_state["hands"]: 
                left_arm_pressed = SystemContext.vr_state.controller_state["hands"]["left"]["buttons"]["grip"] > 0.8          
                left_grip_pressed = SystemContext.vr_state.controller_state["hands"]["left"]["buttons"]["trigger"] > 0.8

            data_collection_bool = (frame is not None) and (right_arm_pressed or left_arm_pressed or right_grip_pressed or left_grip_pressed)

            if data_collection_bool:
                # Generate PCD from RGB-D
                pcd_points = None
                pcd_colors = None
                if frame is not None and depth is not None:
                    try:
                        rgb_frame = frame
                        depth_frame = depth
                        print("camera time stamp : ", frame_stamp)
                        
                        # Determine intrinsics based on depth image size
                        H_d, W_d = depth_frame.shape
                        if W_d == 848 and H_d == 480:
                            intrinsics = REALSENSE_D435_INTRINSICS_848x480
                        else:
                            intrinsics = REALSENSE_D435_INTRINSICS
                        
                        # Resize RGB to match depth if needed
                        if rgb_frame.shape[:2] != depth_frame.shape:
                            try:
                                import cv2
                                rgb_frame = cv2.resize(rgb_frame, (depth_frame.shape[1], depth_frame.shape[0]))
                            except ImportError:
                                logging.warning("[demo_logger] cv2 not available for RGB resize")
                        depth_frame_for_pcd = np.where(depth_frame > 3000, 0, depth_frame)
                        # Convert to point cloud (all points, no downsampling)
                        pcd_points, pcd_colors = rgbd_to_pointcloud(
                            rgb_frame, depth_frame_for_pcd,
                            intrinsics['fx'], intrinsics['fy'],
                            intrinsics['cx'], intrinsics['cy'],
                            max_points=Settings.MAX_POINT
                        )
                        
                        # Debug: check if PCD was generated
                        if pcd_points is not None and len(pcd_points) > 0:
                            print(f"[demo_logger] Generated PCD: {len(pcd_points)} points")
                        else:
                            print(f"[demo_logger] WARNING: PCD generation returned empty result")
                    except Exception as e:
                        logging.warning(f"[demo_logger] Failed to generate PCD: {e}")
                
                print("demo saved\n")
                '''#lerobot data
                data_to_save = {
                    "robot_position": robot_pos,
                    "robot_target_joints": robot_target_joints,
                    "gripper_state": grip,
                    "base_state": base_state,
                    "head_rgb": headcam_sub.curr_frame,
                }
                SystemContext.lerobot_handler.add_frame(data_to_save)'''
                
                
                h5_writer.put({
                    "ts": time.time(),
                    "robot_position": robot_pos,
                    "robot_target_joints": robot_target_joints,
                    "gripper_state": grip,
                    "gripper_target": gripper.get_normalized_target() if gripper else None,
                    "base_state" : base_state,
                    "head_rgb": headcam_sub.curr_frame,
                    "head_rgb_ts": headcam_sub.last_stamp[1] if headcam_sub.last_stamp else None,
                    "head_depth": headcam_sub.curr_depth,
                    "head_depth_ts": headcam_sub.last_depth_stamp[1] if headcam_sub.last_depth_stamp else None,
                    "pcd_points": pcd_points,
                    "pcd_colors": pcd_colors
                })

            # pacing at ~30 FPS with drift correction
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # we overran; reset schedule to now to avoid accumulating lag
                next_t = time.perf_counter()

            # TODO: if stop record is triggered stop 
            # 

    t = threading.Thread(target=_loop, name="demo_logger", daemon=True)
    t.start()
    logging.info(f"Started demo logger at {fps} FPS")
    return stop_event


def start_realsense_camera():
    """Stop existing ROS nodes/topics that may publish camera data, then start the
    RealSense ROS2 camera launch in a subprocess. This function attempts several
    graceful shutdown steps:

    1. Try to gracefully shutdown nodes via lifecycle (if supported).
    2. Kill processes that contain common ROS2 launch/run signatures or the
       'realsense' keyword.
    3. Finally, launch the RealSense node via ros2 launch.

    Returns the subprocess.Popen process for the launched camera, or None on failure.
    """
    def _stop_all_ros_nodes(timeout: float = 3.0):
        """Attempt to stop running ROS2 nodes and camera publishers.

        This helper is best-effort and will not raise on failure; it logs actions.
        """
        # 1) List ROS2 nodes
        try:
            out = subprocess.check_output("ros2 node list", shell=True, stderr=subprocess.DEVNULL, text=True, timeout=2)
            nodes = [ln.strip() for ln in out.splitlines() if ln.strip()]
            logging.info(f"Found ROS2 nodes: {nodes}")
        except Exception as e:
            logging.debug(f"ros2 node list failed: {e}")
            nodes = []

        # 2) Try lifecycle shutdown for lifecycle-enabled nodes
        for n in nodes:
            try:
                # attempt to put node into shutdown (if supports lifecycle)
                subprocess.run(f"ros2 lifecycle set {shlex.quote(n)} shutdown", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
                logging.info(f"Lifecycle shutdown requested for node {n}")
            except Exception:
                pass

        # 3) Small pause to let nodes exit
        time.sleep(min(1.0, timeout))

        # 4) Kill processes that look like ROS2 launch/run or contain 'realsense'
        try:
            ps = subprocess.check_output(['ps', 'aux'], text=True)
        except Exception:
            ps = ''

        killed = []
        for line in ps.splitlines():
            # look for common ros2 invocation patterns or realsense
            if 'ros2' in line or 'realsense' in line or 'rs_launch' in line or 'realsense2_camera' in line:
                try:
                    parts = line.split()
                    pid = int(parts[1])
                    # avoid killing our own process
                    if pid == os.getpid():
                        continue
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed.append(pid)
                    except Exception:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed.append(pid)
                        except Exception:
                            logging.debug(f"Failed to kill pid {pid}")
                except Exception:
                    continue

        if killed:
            logging.info(f"Terminated processes matching ros2/realsense: {killed}")

        # 5) As a final fallback, try pkill for realsense or ros2 launch processes
        try:
            subprocess.run("pkill -f realsense", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run("pkill -f 'ros2 launch'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # attempt to stop existing ROS nodes/topics first
    try:
        logging.info("Stopping any existing ROS2 nodes/topics that may publish camera data...")
        _stop_all_ros_nodes(timeout=3.0)
    except Exception as e:
        logging.warning(f"Failed while attempting to stop existing ROS nodes: {e}")

    # Now launch the RealSense camera via ros2 launch
    try:
        cmd = "bash -c 'source /opt/ros/humble/setup.bash && ros2 launch realsense2_camera rs_launch.py'"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp  # Create new process group
        )
        logging.info(f"Started RealSense camera node (PID: {process.pid})")
        time.sleep(3)  # Wait for camera to initialize
        return process
    except Exception as e:
        logging.error(f"Failed to start RealSense camera: {e}")
        return None


def main(args: argparse.Namespace):
    logging.info("=== VR Control System Starting ===")
    logging.info(f"Server Address       : {args.server}")
    logging.info(f"Local (UPC) IP       : {args.local_ip}:{Settings.vr_control_local_port}")
    logging.info(f"Meta Quest IP        : {args.meta_quest_ip}:{Settings.vr_control_meta_quest_port}")
    logging.info(f"Use Gripper          : {'No' if args.no_gripper else 'Yes'}")
    logging.info(f"RB-Y1 gRPC Address   : {args.rby1}")
    logging.info(f"RB-Y1 Model          : {args.rby1_model}")
    logging.info(f"Use Head             : {'No' if args.no_head else 'Yes'}")

    # Start RealSense camera node
    camera_process = start_realsense_camera()

    rclpy.init()

    def power_off_and_stop():
        rec_data.set()  # stop signal for logging thread
        h5_writer.stop()  # save h5 file and exit
        #SystemContext.lerobot_handler.save_episode()  # save lerobot episode
        robot.power_off(".*")
        # Clean up camera process
        if camera_process is not None:
            try:
                logging.info("Shutting down RealSense camera node...")
                camera_process.terminate()
                camera_process.wait(timeout=5)
            except Exception as e:
                logging.warning(f"Failed to cleanly terminate camera process: {e}")
                camera_process.kill()

    socket = open_zmq_pub_socket(args.server)
    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)
    model = robot.model()
    setup_meta_quest_udp_communication(
        args.local_ip, 
        Settings.vr_control_local_port, 
        args.meta_quest_ip,
        Settings.vr_control_meta_quest_port, 
        power_off_and_stop,
    )

    gripper = None
    if not args.no_gripper:
        for arm in ["left", "right"]:
            if not robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to supply 12V to tool flange. ({arm})")
        time.sleep(0.5)
        gripper = Gripper()
        if not gripper.initialize(verbose=True):
            exit(1)
        gripper.homing()
        gripper.start()
        gripper.set_normalized_target(np.array([0.0, 0.0]))

    pub_thread = threading.Thread(target=publish_gv, args=(socket,), daemon=True)
    pub_thread.start()


    output_path = get_next_h5_path("/media/nvidia/T7/Demo")
    h5_writer = H5Writer(path=output_path, flush_every=60, flush_secs=1.0).start()
    '''#lerobot data handler
    data_handler = LeRobotDataHandler(
        repo_id="rby1_teleop_demo",
        root_dir=output_path/"LeRobotData",
        fps=Settings.rec_fps
    )
    data_handler.initialize_dataset()'''
    
    rec_data = start_demo_logger(gripper, h5_writer, robot, fps=Settings.rec_fps)

    # expose writer and stop-event so button handlers can stop and save
    SystemContext.h5_writer = h5_writer
    #SystemContext.lerobot_handler = data_handler
    SystemContext.rec_stop_event = rec_data

    logging.info(f"output path is {output_path}\n h5 is {h5_writer}\n")

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False
    while True:
        now = time.monotonic()
        if now < next_time:
            time.sleep(next_time - now)
        next_time += Settings.dt

        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[0] = right_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)

            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[1] = 1. - left_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)

        if SystemContext.vr_state.joint_positions.size == 0:
            continue
        
        if handle_vr_button_event(robot, args.no_head):  #실행이 안됨
            if stream is not None:
                stream.cancel()
                stream = None

        if not SystemContext.vr_state.is_initialized:
            continue

        if SystemContext.vr_state.is_stopped:
            if stream is not None:
                stream.cancel()
                stream = None
            SystemContext.vr_state.is_initialized = False
            continue

        logging.info(f"{SystemContext.vr_state.center_of_mass = }")

        dyn_state.set_q(SystemContext.vr_state.joint_positions.copy())
        dyn_robot.compute_forward_kinematics(dyn_state)

        SystemContext.vr_state.base_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx, link_torso_5_idx)
        SystemContext.vr_state.torso_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                     link_torso_5_idx)
        SystemContext.vr_state.right_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                        link_right_arm_6_idx) @ Settings.T_hand_offset
        SystemContext.vr_state.left_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                       link_left_arm_6_idx) @ Settings.T_hand_offset

        trans_12 = dyn_robot.compute_transformation(dyn_state, 1, 2)
        trans_13 = dyn_robot.compute_transformation(dyn_state, 1, 3)
        center = (trans_12[:3, 3] + trans_13[:3, 3]) / 2
        yaw = np.atan2(center[1], center[0])
        pitch = np.atan2(-center[2], center[0]) - np.deg2rad(10)
        yaw = np.clip(yaw, -np.deg2rad(29), np.deg2rad(29))
        pitch = np.clip(pitch, -np.deg2rad(19), np.deg2rad(89))

        # Tracking
        if stream is None:
            if robot.wait_for_control_ready(0):
                stream = robot.create_command_stream()
                SystemContext.vr_state.mobile_linear_velocity = np.array([0.0, 0.0])
                SystemContext.vr_state.mobile_angular_velocity = 0.
                SystemContext.vr_state.is_right_following = False
                SystemContext.vr_state.is_left_following = False
                SystemContext.vr_state.base_start_pose = SystemContext.vr_state.base_pose
                SystemContext.vr_state.torso_locked_pose = SystemContext.vr_state.torso_current_pose
                SystemContext.vr_state.right_hand_locked_pose = SystemContext.vr_state.right_ee_current_pose
                SystemContext.vr_state.left_hand_locked_pose = SystemContext.vr_state.left_ee_current_pose

        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                thumbstick_axis = right_controller["buttons"]["thumbstickAxis"]
                acc = np.array([thumbstick_axis[1], thumbstick_axis[0]])
                SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * acc
                # SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                #                                                   thumbstick_axis[0]
                SystemContext.vr_state.right_controller_current_pose = T_conv.T @ pose_to_se3(
                    right_controller["position"],
                    right_controller["rotation"]) @ T_conv

                trigger_pressed = right_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_right_following and not trigger_pressed:
                    SystemContext.vr_state.is_right_following = False
                if not SystemContext.vr_state.is_right_following and trigger_pressed:
                    SystemContext.vr_state.right_controller_start_pose = SystemContext.vr_state.right_controller_current_pose
                    SystemContext.vr_state.right_ee_start_pose = SystemContext.vr_state.right_ee_current_pose
                    SystemContext.vr_state.is_right_following = True
                    right_reset = True
            else:
                SystemContext.vr_state.is_right_following = False

            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                thumbstick_axis = left_controller["buttons"]["thumbstickAxis"]
                # SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * \
                #                                                  thumbstick_axis[1]
                SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                                                                  thumbstick_axis[0]
                SystemContext.vr_state.left_controller_current_pose = T_conv.T @ pose_to_se3(
                    left_controller["position"],
                    left_controller["rotation"]) @ T_conv

                trigger_pressed = left_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_left_following and not trigger_pressed:
                    SystemContext.vr_state.is_left_following = False
                if not SystemContext.vr_state.is_left_following and trigger_pressed:
                    SystemContext.vr_state.left_controller_start_pose = SystemContext.vr_state.left_controller_current_pose
                    SystemContext.vr_state.left_ee_start_pose = SystemContext.vr_state.left_ee_current_pose
                    SystemContext.vr_state.is_left_following = True
                    left_reset = True
            else:
                SystemContext.vr_state.is_left_following = False

            if "head" in SystemContext.vr_state.controller_state:
                head_controller = SystemContext.vr_state.controller_state["head"]
                SystemContext.vr_state.head_controller_current_pose = T_conv.T @ pose_to_se3(
                    head_controller["position"],
                    head_controller["rotation"]) @ T_conv

                following = SystemContext.vr_state.is_right_following and SystemContext.vr_state.is_left_following and torso_mode
                if SystemContext.vr_state.is_torso_following and not following:
                    SystemContext.vr_state.is_torso_following = False
                if not SystemContext.vr_state.is_torso_following and following:
                    SystemContext.vr_state.head_controller_start_pose = SystemContext.vr_state.head_controller_current_pose
                    SystemContext.vr_state.torso_start_pose = SystemContext.vr_state.torso_current_pose
                    SystemContext.vr_state.is_torso_following = True
                    torso_reset = True
            else:
                SystemContext.vr_state.is_torso_following = False

        SystemContext.vr_state.mobile_linear_velocity -= Settings.mobile_linear_damping_gain * SystemContext.vr_state.mobile_linear_velocity
        SystemContext.vr_state.mobile_angular_velocity -= Settings.mobile_angular_damping_gain * SystemContext.vr_state.mobile_angular_velocity

        if stream:
            try:
                if SystemContext.vr_state.is_right_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.right_controller_start_pose) @ SystemContext.vr_state.right_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.right_ee_start_pose[:3, :3]
                    right_T = SystemContext.vr_state.right_ee_start_pose @ diff_global
                    SystemContext.vr_state.right_hand_locked_pose = right_T
                else:
                    right_T = SystemContext.vr_state.right_hand_locked_pose

                if SystemContext.vr_state.is_left_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.left_controller_start_pose) @ SystemContext.vr_state.left_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.left_ee_start_pose[:3, :3]
                    left_T = SystemContext.vr_state.left_ee_start_pose @ diff_global
                    SystemContext.vr_state.left_hand_locked_pose = left_T
                else:
                    left_T = SystemContext.vr_state.left_hand_locked_pose

                if SystemContext.vr_state.is_torso_following and torso_mode:
                    print('a')
                    diff = np.linalg.inv(
                        SystemContext.vr_state.head_controller_start_pose) @ SystemContext.vr_state.head_controller_current_pose

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.torso_start_pose[:3, :3]
                    torso_T = SystemContext.vr_state.torso_start_pose @ diff
                    SystemContext.vr_state.torso_locked_pose = torso_T
                else:
                    torso_T = SystemContext.vr_state.torso_locked_pose

                if args.whole_body:
                    ctrl_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                        .set_joint_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset | left_reset | torso_reset)
                    )
                    ctrl_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                else:
                    torso_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6)
                        .set_joint_torque_limit([500] * 6)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(torso_reset)
                    )
                    right_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset)
                    )
                    left_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(left_reset)
                    )
                    torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    right_builder.add_target("base", "link_right_arm_6",
                                             right_T @ np.linalg.inv(Settings.T_hand_offset),
                                             2, np.pi * 2, 20, np.pi * 80)
                    left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(torso_builder)
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                torso_reset = False
                right_reset = False
                left_reset = False

                stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder()
                        # .set_head_command(
                        #     rby.JointPositionCommandBuilder()
                        #     .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        #     .set_position([float(yaw), float(pitch)])
                        #     .set_minimum_time(Settings.dt * 1.01)
                        # )
                        .set_mobility_command(
                            rby.SE2VelocityCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                            .set_velocity(-SystemContext.vr_state.mobile_linear_velocity,
                                          -SystemContext.vr_state.mobile_angular_velocity)
                            .set_minimum_time(Settings.dt * 1.01)
                        )
                        .set_body_command(
                            ctrl_builder
                        )
                    )
                )
            except Exception as e:
                logging.error(e)
                stream = None
                exit(1)

        # ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RB-Y1 VR Control Launcher")

    parser.add_argument(
        "-s", "--server", type=str, default="tcp://*:5555",
        help="ZMQ server address for the UPC (default: tcp://*:5555)"
    )
    parser.add_argument(
        "--local_ip", required=True, type=str,
        help="Local Wi-Fi (or LAN) IP address of the UPC"
    )
    parser.add_argument(
        "--meta_quest_ip", required=True, type=str,
        help="Wi-Fi (or LAN) IP address of the Meta Quest"
    )
    parser.add_argument(
        "--no_gripper", action="store_true",
        help="Run without gripper support"
    )
    parser.add_argument(
        "--rby1", default="192.168.30.1:50051", type=str,
        help="gRPC address of the RB-Y1 robot (default: 192.168.30.1:50051)"
    )
    parser.add_argument(
        "--rby1_model", default="a", type=str,
        help="Model type of the RB-Y1 robot (default: a)"
    )
    parser.add_argument(
        "--no_head", action="store_false", 
        help="Run without controlling the head"
    )
    parser.add_argument(
        "--whole_body", action="store_true",
        help="Use a whole-body optimization formulation (single control for all joints)"
    )

    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
