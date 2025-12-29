import copy
import threading
import logging
import numpy as np
import subprocess
import shlex
import os
import signal
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from utils import rosimg_to_numpy

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
