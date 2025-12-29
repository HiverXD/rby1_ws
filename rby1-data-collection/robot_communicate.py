import logging
import rby1_sdk as rby
from setup import SystemContext, Settings

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
