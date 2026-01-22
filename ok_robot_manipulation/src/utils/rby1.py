"""
RBY1 Robot utilities for grasp execution and control.
Handles robot connection, motion planning, and pose computation.
"""

import sys
import json
import socket
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rby1_sdk as rby


class RemoteGripperClient:
    """Simple UDP client to send gripper commands to a remote host."""

    def __init__(self, host: str, port: int = 5009, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def close(self, position: float = 0.8, hold_time: float = 1.0) -> bool:
        payload = {
            "cmd": "close",
            "position": float(position),
            "hold_time": float(hold_time),
        }
        data = json.dumps(payload).encode("utf-8")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout)
                sock.sendto(data, (self.host, self.port))
            print(f"[INFO] Sent remote gripper close command to {self.host}:{self.port} (position={position}, hold={hold_time}s)")
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to send remote gripper command: {exc}")
            return False


def connect_to_robot(robot_address):
    """Connect to robot and return robot and model instances."""
    print(f"[INFO] Connecting to robot at {robot_address}...")
    
    # Create robot instance
    robot = rby.create_robot_a(robot_address)
    model = robot.model()
    
    if not robot.connect():
        print("[ERROR] Could not connect to robot")
        sys.exit(1)
    
    print("[INFO] Connected to robot successfully")
    
    # Initialize robot control system
    print("[INFO] Initializing robot control system...")
    
    # Turn on power
    print("[INFO] Turning on power...")
    robot.power_on(".*")
    
    # Enable servo motors
    print("[INFO] Enabling servo motors...")
    robot.servo_on(".*")
    
    # Reset fault control manager
    print("[INFO] Resetting fault control manager...")
    robot.reset_fault_control_manager()
    
    # Enable control manager
    print("[INFO] Enabling control manager...")
    robot.enable_control_manager(unlimited_mode_enabled=True)
    
    print("[INFO] Robot control system initialized successfully")
    
    return robot, model


def move_to_zero_position(robot, minimum_time=5.0):
    """Move robot to zero position (home position).
    
    Args:
        robot: Robot instance
        minimum_time: Time for motion (seconds)
    """
    print(f"[INFO] Moving body to zero position...")
    
    rc_builder = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position([0.] * 6)
            )
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position([0.] * 7)
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position([0.] * 7)
            )
        )
    )
    
    feedback = robot.send_command(rc_builder).get()
    if feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok:
        print("[INFO] Body moved to zero position")
    else:
        print(f"[WARN] Failed to move to zero position: {feedback.finish_code}")
        return False

    print("[INFO] Moving head to zero position...")
    rc_builder = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_head_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
            .set_minimum_time(minimum_time)
            .set_position(np.deg2rad([0.0, 0.0]))  # [joint_head_1, joint_head_2] - zero position
        )
    )
    
    feedback = robot.send_command(rc_builder).get()
    if feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok:
        print("[INFO] Head moved to zero position")
        return True
    else:
        print(f"[WARN] Failed to move head to zero position: {feedback.finish_code}")
        return False


def move_to_ready_position(robot, minimum_time=3.0):
    """Move robot to ready position for grasping.
    
    Ready pose is a stable configuration with arms positioned for manipulation:
    - Torso: [0, 45, -90, 45, 0, 0] degrees
    - Right arm: [0, -5, 0, -120, 0, 70, 0] degrees  
    - Left arm: [0, 5, 0, -120, 0, 70, 0] degrees
    - Head: [0, 90] degrees (looking down at maximum)
    
    Args:
        robot: Robot instance
        minimum_time: Time for motion (seconds)
    """
    print(f"[INFO] Moving to ready position...")
    
    # Move body first
    rc_builder = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
                .set_minimum_time(minimum_time)
                .set_position(np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]))
            )
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
                .set_minimum_time(minimum_time)
                .set_position(np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]))
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
                .set_minimum_time(minimum_time)
                .set_position(np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]))
            )
        )
    )
    
    feedback = robot.send_command(rc_builder).get()
    if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        print(f"[WARN] Failed to move body to ready position: {feedback.finish_code}")
        return False
    
    print("[INFO] Body moved to ready position")
    
    # Move head to look down
    print("[INFO] Tilting head down to look at workspace...")
    rc_builder = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_head_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
            .set_minimum_time(minimum_time)
            .set_position(np.deg2rad([0.0, 45.0]))  # [joint_head_1, joint_head_2]
        )
    )
    
    feedback = robot.send_command(rc_builder).get()
    if feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok:
        print("[INFO] Head tilted down successfully")
        return True
    else:
        print(f"[WARN] Failed to tilt head: {feedback.finish_code}")
        return False


def get_camera_pose_from_robot(robot, model, head_camera_link):
    """Compute camera pose relative to base link using current robot state.
    
    The camera is offset from head_link_2:
    - Translation: +0.06m in Z axis of head_link_2
    - Rotation: head_link_2's +X is camera's +Z (forward), 
                head_link_2's +Z is camera's -Y (up), 
                head_link_2's -Y is camera's +X (right)
    """
    # Get dynamics model
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(["base", head_camera_link], model.robot_joint_names)
    BASE_INDEX, HEAD_CAMERA_INDEX = 0, 1
    
    # Get current robot state
    robot_state = robot.get_state()
    dyn_state.set_q(robot_state.position)
    
    # Compute forward kinematics
    dyn_model.compute_forward_kinematics(dyn_state)
    
    # Get transformation from base to head_link_2
    T_base2head = dyn_model.compute_transformation(dyn_state, BASE_INDEX, HEAD_CAMERA_INDEX)
    
    print(f"\n[DEBUG] Head Link Transformation Matrix (base -> {head_camera_link}):")
    print(T_base2head)
    
    # Define transformation from head_link_2 to camera
    # Translation:
    # Rotation: Transform from head_link_2 frame to camera frame
    # head_link_2: +X forward, +Z up, -Y right
    # camera: +X right, +Y down, +Z forward
    # Rotation matrix columns are the camera axes expressed in head_link_2 frame:
    # Camera +X (right) = head_link_2 -Y: [0, -1, 0]
    # Camera +Y (down) = head_link_2 -Z: [0, 0, -1]
    # Camera +Z (forward) = head_link_2 +X: [1, 0, 0]
    T_head2camera = np.array([
        [0, 0, 1, 0.0205 - 0.0042],
        [-1, 0, 0, -0.0115],
        [0, -1, 0, 0.040 + 0.0125],
        [0, 0, 0, 1]
    ])
    
    # Compute combined transformation
    T_base2camera = T_base2head @ T_head2camera
    
    print(f"\n[DEBUG] Camera Transformation Matrix (base -> camera):")
    print(T_base2camera)
    print(f"[DEBUG] Camera X-axis (right): {T_base2camera[0:3, 0]}")
    print(f"[DEBUG] Camera Y-axis (down): {T_base2camera[0:3, 1]}")
    print(f"[DEBUG] Camera Z-axis (forward): {T_base2camera[0:3, 2]}")
    print(f"[DEBUG] Camera Position: {T_base2camera[0:3, 3]}\n")
    
    # Extract position
    camera_pos = T_base2camera[0:3, 3]
    
    # Extract rotation matrix and convert to quaternion (w, x, y, z)
    rotation_matrix = T_base2camera[0:3, 0:3]
    rot = R.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
    camera_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]
    
    print(f"[INFO] Camera pose - Position: {camera_pos}")
    print(f"[INFO] Camera pose - Quaternion (w,x,y,z): {camera_quat}")
    
    return [camera_pos, camera_quat]


def print_pose_error(robot, model, ee_link, desired_T, step_name):
    """Helper function to print current pose and error."""
    # Get current robot state
    robot_state = robot.get_state()
    
    # Compute forward kinematics
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(["base", ee_link], model.robot_joint_names)
    dyn_state.set_q(robot_state.position)
    dyn_model.compute_forward_kinematics(dyn_state)
    
    # Get current end-effector pose
    T_current = dyn_model.compute_transformation(dyn_state, 0, 1)  # base to ee
    
    # Extract position and rotation
    current_pos = T_current[0:3, 3]
    current_rot = T_current[0:3, 0:3]
    current_quat_xyzw = R.from_matrix(current_rot).as_quat()
    current_quat_wxyz = np.array([current_quat_xyzw[3], current_quat_xyzw[0], 
                                   current_quat_xyzw[1], current_quat_xyzw[2]])
    
    # Compute errors
    desired_pos = desired_T[0:3, 3]
    desired_rot = desired_T[0:3, 0:3]
    
    pos_error = np.linalg.norm(current_pos - desired_pos)
    
    # Rotation error (angle of rotation difference)
    R_error = desired_rot.T @ current_rot
    rotation_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
    
    print(f"\n[{step_name} - Current Pose]")
    print(f"  Position: {current_pos}")
    print(f"  Quaternion (w,x,y,z): {current_quat_wxyz}")
    print(f"[{step_name} - Desired Pose]")
    print(f"  Position: {desired_pos}")
    print(f"[{step_name} - Error]")
    print(f"  Position error: {pos_error*1000:.2f} mm")
    print(f"  Rotation error: {np.rad2deg(rotation_error):.2f} deg\n")


def execute_grasp_motion(
    robot,
    model,
    target_pos,
    target_quat_wxyz,
    arm_side,
    pre_grasp_offset=0.1,
    gripper=None,
    gripper_close_position=0.8,
    gripper_hold_time=1.0,
    remote_gripper_host=None,
    remote_gripper_port=5009,
    remote_gripper_timeout=2.0,
):
    """Execute grasp using OptimalControlCommandBuilder for accurate IK-based motion.
    
    Args:
        robot: Robot instance
        model: Robot model
        target_pos: Target position [x, y, z] in world frame
        target_quat_wxyz: Target orientation quaternion [w, x, y, z]
        arm_side: 'left' or 'right'
        pre_grasp_offset: Distance to approach before closing gripper (meters)
        gripper: Gripper instance (from DynamixelBus) for real robot control (optional)
        gripper_close_position: Normalized gripper position 0.0-1.0 (default: 0.8)
        gripper_hold_time: Time to hold gripper closed in seconds (default: 1.0)
        remote_gripper_host: If provided, send gripper close command to remote host via UDP
        remote_gripper_port: UDP port for remote gripper host (default: 5009)
        remote_gripper_timeout: UDP send timeout in seconds
    """
    
    print(f"\n[INFO] Executing grasp with {arm_side} arm...")
    
    # Wait for control to be ready
    print("[INFO] Waiting for control system to be ready...")
    if not robot.wait_for_control_ready(100):
        print("[ERROR] Control system not ready within timeout")
        return False
    
    # Convert quaternion to rotation matrix
    rot = R.from_quat([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]])  # [x,y,z,w]
    target_rot_matrix = rot.as_matrix()
    
    # Create target transformation matrix
    T_target = np.eye(4)
    T_target[0:3, 0:3] = target_rot_matrix
    approach_vector = -target_rot_matrix[:, 2]  # Z-axis of gripper frame
    # T_target[0:3, 3] = target_pos
    T_target[0:3, 3] = target_pos - approach_vector * 0.14  # Adjust for gripper length (~14cm)
    
    print(f"[DEBUG] Target position: {target_pos}")
    print(f"[DEBUG] Target orientation (wxyz): {target_quat_wxyz}")
    
    # Determine arm parameters
    if arm_side == 'right':
        ee_link = "ee_right"
    else:
        ee_link = "ee_left"
    
    def get_current_ee_pose():
        """Get current end-effector pose."""
        robot_state = robot.get_state()
        dyn_model = robot.get_dynamics()
        dyn_state = dyn_model.make_state(["base", ee_link], model.robot_joint_names)
        dyn_state.set_q(robot_state.position)
        dyn_model.compute_forward_kinematics(dyn_state)
        return dyn_model.compute_transformation(dyn_state, 0, 1)
    
    def ensure_control_manager_enabled():
        """Ensure control manager is enabled and ready."""
        robot.reset_fault_control_manager()
        robot.enable_control_manager(unlimited_mode_enabled=True)
        if not robot.wait_for_control_ready(100):
            print("[ERROR] Control system not ready within timeout")
            return False
        return True
    
    def move_arm_to_pose(T_desired, description=""):
        """Move arm to desired pose using CartesianCommandBuilder.
        
        Args:
            T_desired: Target 4x4 transformation matrix
            description: Description for logging
        """
        print(f"[INFO] {description}")
        
        # Ensure control manager is enabled before motion
        if not ensure_control_manager_enabled():
            return False
        
        # Get current pose for logging
        T_current = get_current_ee_pose()
        print(f"[DEBUG] Current position: {T_current[0:3, 3]}")
        print(f"[DEBUG] Target position: {T_desired[0:3, 3]}")
        
        # Build CartesianCommand for the arm
        LINEAR_VELOCITY_LIMIT = 0.1  # m/s
        ANGULAR_VELOCITY_LIMIT = np.pi * 0.1  # rad/s
        ACCELERATION_LIMIT = 0.8
        MINIMUM_TIME = 5.0
        
        cartesian_command = (
            rby.CartesianCommandBuilder()
            .add_target("base", ee_link, T_desired,
                       LINEAR_VELOCITY_LIMIT,
                       ANGULAR_VELOCITY_LIMIT,
                       ACCELERATION_LIMIT)
            .set_minimum_time(MINIMUM_TIME)
            .set_stop_position_tracking_error(1e-2)
            .set_stop_orientation_tracking_error(5e-2)
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(3.0))
        )
        
        # Wrap in arm command based on arm side
        body_builder = rby.BodyComponentBasedCommandBuilder()
        if arm_side == 'right':
            body_builder.set_right_arm_command(cartesian_command)
        else:
            body_builder.set_left_arm_command(cartesian_command)
        
        rc_builder = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body_builder)
        )
        
        # Send command (priority=1 is default)
        try:
            feedback = robot.send_command(rc_builder, 1).get()
            if feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok:
                print(f"[INFO] Motion completed successfully")
            else:
                print(f"[WARN] Motion finished with: {feedback.finish_code}")
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return False
        
        # Check final position error
        T_final = get_current_ee_pose()
        pos_error = np.linalg.norm(T_final[0:3, 3] - T_desired[0:3, 3])
        print(f"[INFO] Final position error: {pos_error*1000:.1f} mm")
        
        return True
    
    # Step 1: Move to pre-grasp pose (offset along approach direction)
    print(f"[Step 1/4] Moving to pre-grasp position (offset: {pre_grasp_offset} + 0.14m)...")
    # approach_vector = -target_rot_matrix[:, 2]  # Z-axis of gripper frame
    pre_grasp_pos = target_pos - approach_vector * (pre_grasp_offset + 0.14)
    
    print(f"[DEBUG] Pre-grasp position: {pre_grasp_pos}")
    print(f"[DEBUG] Approach vector: {approach_vector}")
    
    T_pre_grasp = np.eye(4)
    T_pre_grasp[0:3, 0:3] = target_rot_matrix
    T_pre_grasp[0:3, 3] = pre_grasp_pos
    
    if not move_arm_to_pose(T_pre_grasp, description="Moving to pre-grasp..."):
        print(f"[ERROR] Pre-grasp motion failed")
        return False
    
    # Print current pose and error
    print_pose_error(robot, model, ee_link, T_pre_grasp, "Step 1: Pre-grasp")
    
    # Wait for user confirmation before proceeding
    input("\n✋ [STEP 1 COMPLETE] Press ENTER to proceed to Step 2 (Grasp Approach)...\n")

    # Step 2: Move to grasp pose
    print(f"[Step 2/4] Moving to grasp position...")
    
    if not move_arm_to_pose(T_target, description="Moving to grasp position..."):
        print(f"[ERROR] Grasp approach failed")
        return False
    
    # Print current pose and error
    print_pose_error(robot, model, ee_link, T_target, "Step 2: Grasp")
    
    # Wait for user confirmation before proceeding
    input("\n✋ [STEP 2 COMPLETE] Press ENTER to proceed to Step 3 (Close Gripper)...\n")

    # Step 3: Close gripper
    if gripper is not None:
        print(f"[Step 3/4] Closing gripper (position: {gripper_close_position})...")
        gripper.set_target(np.array([gripper_close_position, gripper_close_position]))
        time.sleep(gripper_hold_time)
        print(f"[Step 3/4] Gripper closed")
    elif remote_gripper_host is not None:
        print(f"[Step 3/4] Sending remote gripper close (position: {gripper_close_position}) to {remote_gripper_host}:{remote_gripper_port}...")
        client = RemoteGripperClient(remote_gripper_host, remote_gripper_port, remote_gripper_timeout)
        client.close(position=gripper_close_position, hold_time=gripper_hold_time)
        # Allow remote side to actuate
        time.sleep(gripper_hold_time)
        print(f"[Step 3/4] Remote gripper command sent")
    else:
        # Fallback: manual/simulator gripper closing
        print(f"[Step 3/4] Close gripper manually now...")
        time.sleep(3.0)  # Give time to manually close gripper
    
    # Wait for user confirmation before proceeding
    input("\n✋ [STEP 3 COMPLETE] Press ENTER to proceed to Step 4 (Lift Object)...\n")
    
    # Step 4: Lift object
    print(f"[Step 4/4] Lifting object...")
    lift_offset = 0.10  # Lift 10cm
    T_lift = np.copy(T_target)
    T_lift[2, 3] += lift_offset  # Add to Z coordinate
    
    if not move_arm_to_pose(T_lift, description="Lifting object..."):
        print(f"[ERROR] Lift motion failed")
        return False
    
    # Print current pose and error
    print_pose_error(robot, model, ee_link, T_lift, "Step 4: Lift")
    
    print("[SUCCESS] Grasp executed successfully!")
    return True
