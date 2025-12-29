import logging
import threading
import rby1_sdk as rby
import socket
from typing import Union
import json
import numpy as np
import numpy as np
from setup import SystemContext, Settings
from helper import *
import threading

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

started = False
torso_mode = False  # 글로벌 상태로 변경

def handle_vr_button_event(robot: Union[rby.Robot_A, rby.Robot_M], no_head: bool):
    global started, torso_mode
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

            started = True

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
            if not started:
                movej(
                robot,
                np.zeros(torso_dof),
                Settings.right_arm_midpoint1,
                Settings.left_arm_midpoint1,
                np.zeros(head_dof),
                minimum_time=10,
                )
                started = True

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
        return False, torso_mode

    SystemContext.vr_state.event_right_a_pressed = False
    SystemContext.vr_state.event_right_b_pressed = False
    SystemContext.vr_state.event_left_a_pressed = False
    SystemContext.vr_state.event_left_b_pressed = False

    return True, torso_mode