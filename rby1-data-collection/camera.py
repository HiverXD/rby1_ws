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

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

from message_filters import Subscriber, ApproximateTimeSynchronizer


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

# 여러 개 카메라 사용 시
class MultiCamSub(Node):
    def __init__(self, camera_id: str = "camera"):  # 카메라 ID 파라미터 추가
        super().__init__(f'head_cam_sub_{camera_id}')  # 노드명 유니크하게
        self.camera_id = camera_id
        self._got_first_color = threading.Event()
        self._got_first_depth = threading.Event()
        self._seq = 0

        # 토픽 경로를 동적으로 설정
        color_topic = f'/{camera_id}/camera/color/image_raw'
        depth_topic = f'/{camera_id}/camera/depth/image_rect_raw'
        
        self.color_sub = self.create_subscription(
            Image, color_topic, self.color_cb, qos_profile_sensor_data
        )
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, qos_profile_sensor_data,
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

    def depth_cb(self, msg: Image):
        # Depth image is typically uint16 Z16 format
        depth_arr = np.frombuffer(msg.data, dtype=np.uint16)
        depth_arr = depth_arr.reshape((msg.height, msg.width))
        #with self._lock:
        self.curr_depth = depth_arr
        self.last_depth_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._got_first_depth.set()

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

# ---------- Utilities ----------
def stamp_to_float(stamp) -> float:
    # stamp: builtin_interfaces.msg.Time
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ros_depth_to_numpy(msg: Image) -> np.ndarray:
    # Z16 uint16 depth
    arr = np.frombuffer(msg.data, dtype=np.uint16)
    return arr.reshape((msg.height, msg.width))

@dataclass
class RGBDFrame:
    t: float                        # timestamp in seconds
    stamp: Tuple[int, int]           # (sec, nanosec)
    color: np.ndarray                # HxWx3 uint8
    depth: np.ndarray                # HxW uint16
    seq: int


# ---------- Per-camera RGBD synchronizer ----------
class RGBDSync:
    """
    카메라 1대의 color+depth를 message_filters로 time-sync해서 RGBDFrame을 만든다.
    만든 프레임은 manager 콜백으로 전달한다.
    """
    def __init__(
        self,
        node: Node,
        cam_id: str,
        color_topic: str,
        depth_topic: str,
        on_rgbd: Callable[[str, RGBDFrame], None],
        rgbd_slop: float = 0.05,      # FPS 10(100ms) 기준: 30~60ms 사이에서 상황에 맞게
        rgbd_queue: int = 20
    ):
        self.node = node
        self.cam_id = cam_id
        self.on_rgbd = on_rgbd
        self._seq = 0

        self.color_sub = Subscriber(node, Image, color_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(node, Image, depth_topic, qos_profile=qos_profile_sensor_data)

        self.sync = ApproximateTimeSynchronizer(
            fs=[self.color_sub, self.depth_sub],
            queue_size=rgbd_queue,
            slop=rgbd_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._rgbd_cb)

        node.get_logger().info(
            f"[RGBDSync:{cam_id}] color={color_topic}, depth={depth_topic}, slop={rgbd_slop}, q={rgbd_queue}"
        )

    def _rgbd_cb(self, color_msg: Image, depth_msg: Image):
        # 변환
        color = rosimg_to_numpy(color_msg)              # HxWx3 uint8 (가정)
        depth = ros_depth_to_numpy(depth_msg)           # HxW uint16

        t = stamp_to_float(color_msg.header.stamp)
        stamp = (color_msg.header.stamp.sec, color_msg.header.stamp.nanosec)

        self._seq += 1
        frame = RGBDFrame(t=t, stamp=stamp, color=color, depth=depth, seq=self._seq)

        # manager로 전달
        self.on_rgbd(self.cam_id, frame)


# ---------- Multi-camera synchronizer ----------
class MultiCamRGBDSync(Node):
    """
    3대 카메라의 RGBDFrame을 timestamp 기준으로 "같은 시각 세트"로 매칭한다.
    """
    def __init__(
        self,
        cam_ids: List[str],
        topic_pattern_color: str = "/{cam}/camera/color/image_raw",
        topic_pattern_depth: str = "/{cam}/camera/depth/image_rect_raw",
        rgbd_slop: float = 0.05,          # 각 카메라 내부 RGB-D sync 허용오차
        rgbd_queue: int = 20,
        multicam_tol: float = 0.06,       # 카메라 간 동기 매칭 허용오차 (FPS10이면 50~80ms부터 시도 권장)
        max_buf: int = 50,                # 카메라별 버퍼 최대 (10fps면 5초치=50)
    ):
        super().__init__("multi_cam_rgbd_sync")

        assert len(cam_ids) == 3, "요청 조건: 카메라 3대"

        self.cam_ids = cam_ids
        self.multicam_tol = multicam_tol
        self.max_buf = max_buf

        self._lock = threading.Lock()
        self._buf: Dict[str, deque[RGBDFrame]] = {cid: deque() for cid in cam_ids}

        self._got_first_set = threading.Event()
        self._last_set: Optional[Dict[str, RGBDFrame]] = None
        self._set_seq = 0

        # 카메라별 RGB-D sync 생성
        self._per_cam = []
        for cid in cam_ids:
            c_topic = topic_pattern_color.format(cam=cid)
            d_topic = topic_pattern_depth.format(cam=cid)
            self._per_cam.append(
                RGBDSync(
                    node=self,
                    cam_id=cid,
                    color_topic=c_topic,
                    depth_topic=d_topic,
                    on_rgbd=self._on_rgbd_frame,
                    rgbd_slop=rgbd_slop,
                    rgbd_queue=rgbd_queue,
                )
            )

        self.get_logger().info(
            f"[MultiCam] cams={cam_ids}, multicam_tol={multicam_tol}s, max_buf={max_buf}"
        )

    # ----- Public API -----
    def get_synced_set_copy(self):
        """
        가장 최근에 완성된 '동일 시각' 3카메라 세트를 복사본으로 반환.
        return: (dict(cam_id->(color, depth, stamp, seq)), set_seq) or (None, None)
        """
        with self._lock:
            if self._last_set is None:
                return None, None

            out = {}
            for cid, fr in self._last_set.items():
                out[cid] = (
                    fr.color.copy(),
                    fr.depth.copy(),
                    fr.stamp,
                    fr.seq,
                    fr.t,
                )
            return out, self._set_seq

    def on_synced_rgbd_set(self, synced: Dict[str, RGBDFrame], set_seq: int):
        """
        필요하면 여기 오버라이드/확장해서 '동기 세트'가 만들어질 때마다 처리하면 됨.
        기본은 로그만.
        """
        # 예: timestamp 출력
        ts = {cid: synced[cid].t for cid in self.cam_ids}
        self.get_logger().info(f"[SYNCED #{set_seq}] t={ts}")

    # ----- Internal -----
    def _on_rgbd_frame(self, cam_id: str, frame: RGBDFrame):
        with self._lock:
            q = self._buf[cam_id]
            q.append(frame)
            # 버퍼 크기 제한
            while len(q) > self.max_buf:
                q.popleft()

            # 새 프레임이 들어올 때마다 매칭 시도
            matched = self._try_match_locked()

        if matched is not None:
            synced, set_seq = matched
            self._got_first_set.set()
            # 락 밖에서 user hook 호출
            self.on_synced_rgbd_set(synced, set_seq)

    def _try_match_locked(self) -> Optional[Tuple[Dict[str, RGBDFrame], int]]:
        """
        락이 잡힌 상태에서 호출.
        버퍼들에서 '같은 시각' 세트를 찾으면 pop하면서 반환.
        """
        # 3개 모두 최소 1개씩 있어야 시도 가능
        if any(len(self._buf[cid]) == 0 for cid in self.cam_ids):
            return None

        tol = self.multicam_tol

        # 전략:
        # - 기준 시간 후보를 "각 버퍼의 가장 최신(tail)들 중 가장 작은 값"으로 잡으면
        #   다른 카메라가 그 시간대 프레임을 보유할 가능성이 높음.
        latest_times = [self._buf[cid][-1].t for cid in self.cam_ids]
        t_ref = min(latest_times)

        # 각 카메라 버퍼에서 t_ref에 가장 가까운 프레임을 찾는다.
        chosen_idx = {}
        chosen = {}

        for cid in self.cam_ids:
            q = self._buf[cid]
            # 단순 선형 탐색(버퍼 작아서 충분)
            best_i = None
            best_dt = 1e9
            for i, fr in enumerate(q):
                dt = abs(fr.t - t_ref)
                if dt < best_dt:
                    best_dt = dt
                    best_i = i
            if best_i is None or best_dt > tol:
                # 아직 충분히 맞는 프레임이 없음 → 너무 오래된 프레임 정리하고 다음 기회
                self._drop_too_old_locked(t_ref)
                return None

            chosen_idx[cid] = best_i
            chosen[cid] = q[best_i]

        # 여기까지 왔으면 3개 모두 tol 안에 들어오는 프레임을 찾음
        # 선택된 프레임 이전(및 해당)까지 pop 해서 버퍼 진행
        for cid in self.cam_ids:
            q = self._buf[cid]
            # chosen_idx까지 제거
            for _ in range(chosen_idx[cid] + 1):
                q.popleft()

        # 최신 세트 저장
        self._set_seq += 1
        set_seq = self._set_seq
        self._last_set = chosen

        return chosen, set_seq

    def _drop_too_old_locked(self, t_ref: float):
        """
        t_ref보다 너무 뒤처진 프레임들을 적당히 정리(버퍼 폭발/지연 방지)
        """
        # tol보다 훨씬 작은 시간들은 버리자 (여유 계수 2.0)
        cutoff = t_ref - 2.0 * self.multicam_tol
        for cid in self.cam_ids:
            q = self._buf[cid]
            while len(q) > 0 and q[0].t < cutoff:
                q.popleft()

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
