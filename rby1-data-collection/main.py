import argparse
import logging
import zmq
import time
import threading
import rby1_sdk as rby
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from gripper import Gripper
import pickle
from lerobot_handler import LeRobotDataHandler
from rclpy.executors import SingleThreadedExecutor  # or MultiThreadedExecutor

# demo writer 
from h5py_writer import H5Writer

# ROS2 Camera subscriber
import rclpy

import numpy as np
import time
import yaml

from utils import *

import threading
from helper import * 

from camera import HeadCamSub, start_realsense_camera
from setup import Settings, SystemContext
from robot_communicate import robot_state_callback, connect_rby1
from vr_communicate import setup_meta_quest_udp_communication, handle_vr_button_event

# lerobot dataset의 action 및 observation 크기 정의를 위한 초기 샘플 데이터

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])

def open_zmq_pub_socket(bind_address: str) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)
    logging.info(f"ZMQ PUB server opened at {bind_address}")
    return socket

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

def get_config():
    with open('rby1-data-collection/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config

# NOTE
def start_demo_logger(gripper: Gripper | None, h5_writer, data_handler, robot, fps: int = 30) -> threading.Event:
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

    # headcam_sub = MultiCamSub(camera_id="head")
    # right_wrist_sub = MultiCamSub(camera_id="right_wrist")
    # left_wrist_sub = MultiCamSub(camera_id="left_wrist")

    executor = SingleThreadedExecutor()
    executor.add_node(headcam_sub)
    # executor.add_node(right_wrist_sub)
    # executor.add_node(left_wrist_sub)

    spin_thread = threading.Thread(target=executor.spin, name="ros2_spin", daemon=True)
    spin_thread.start()


    def _loop():
        logging.info("loop started")
        next_t = time.perf_counter()

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
            data_handler.update_previous_target(robot_pos)
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
                #lerobot data
                data_to_save = {
                    "robot_position": robot_pos,
                    "robot_target_joints": robot_target_joints,
                    "gripper_state": grip,
                    "base_state": base_state,
                    "head_rgb": headcam_sub.curr_frame,
                }
                # print("robot_pose shape: ", robot_pos.shape)
                # print("robot_target_joints shape: ", robot_target_joints.shape)
                # print("gripper_state shape: ", grip.shape if grip is not None else None)
                # print("base_state shape: ", base_state.shape if base_state is not None else None)
                # print("head_rgb shape: ", headcam_sub.curr_frame.shape if headcam_sub.curr_frame is not None else None)
                
                data_handler.put(data_to_save)  # 실행이 안됨ㅜㅜㅜ

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

    '''power off and stop에서 data_handler.stop() 정의하기 위해 
    output_path ~ data_handler까지 위치 옮김'''
    # start writing
    
    config = get_config()
    root_path = os.path.join(config['demo_root'], config['task_name'])

    # 디렉토리가 없으면 생성 (이미 있으면 아무 동작 안 함)
    os.makedirs(root_path, exist_ok=True)

    output_path = get_next_h5_path(root_path)
    h5_writer = H5Writer(path=output_path, flush_every=60, flush_secs=1.0).start()
    #lerobot data handler
    data_handler = LeRobotDataHandler(
        repo_id="rby1_teleop_demo",
        root_dir=get_next_lerobot_path("/media/nvidia/T7/Demo/LeRobotData"),
        fps=Settings.rec_fps
    )
    
    
    data_handler.start()
    
    def power_off_and_stop():
        rec_data.set()  # stop signal for logging thread
        h5_writer.stop()  # save h5 file and exit
        data_handler.stop()  # save lerobot episode
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

    rec_data = start_demo_logger(gripper, h5_writer, data_handler, robot, fps=Settings.rec_fps)

    logging.info("data handler run started")

    # data_handler._run()

    # expose writer and stop-event so button handlers can stop and save
    SystemContext.h5_writer = h5_writer
    SystemContext.lerobot_handler = data_handler
    SystemContext.rec_stop_event = rec_data

    logging.info(f"output path is {output_path}\n h5 is {h5_writer}\n")
    logging.info(f"lerobot handler is {data_handler}\n")

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False

    logging.info("Entering main control loop")

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
        button_event, torso_mode = handle_vr_button_event(robot, args.no_head)
        
        if button_event:  #실행이 안됨
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
                logging.error(f"Stream error: {e}")
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
