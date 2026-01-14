import h5py
import queue
from typing import Optional
import time
import threading
import numpy as np
import os
from pathlib import Path
import logging

class H5Writer:
    def __init__(self, path: str, flush_every: int = 60, flush_secs: float = 1.0, save_pcd: bool = True):
        self.path = path
        self.flush_every = flush_every
        self.flush_secs = flush_secs
        self.save_pcd = save_pcd
        self.q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._update_queue: "queue.Queue[tuple]" = queue.Queue()  # (target_idx, robot_pos) 튜플 저장
        self._current_idx = 0  # 현재까지 저장된 샘플 수 추적

    def start(self):
        print("H5 writer start\n")
        self.thread = threading.Thread(target=self._run, name="h5_writer", daemon=True)
        self.thread.start()
        return self

    def stop(self, join: bool = True):
        self.stop_event.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if join and self.thread:
            self.thread.join()

    def update_previous_target(self, robot_pos: np.ndarray):
        """이전 스텝의 robot_target_joints를 현재 robot_pos로 업데이트
        
        Args:
            robot_pos: 현재 robot_pos 값
        """
        try:
            # 현재 저장된 샘플 수를 기준으로 이전 스텝의 인덱스 계산
            # 현재 스텝이 저장되기 전이므로, 이전 스텝의 인덱스는 _current_idx - 1
            target_idx = max(0, self._current_idx - 1)  # 최소 0
            self._update_queue.put_nowait((target_idx, robot_pos))
        except queue.Full:
            pass
    
    def put(self, sample: dict, drop_oldest: bool = True):
        """
        Expected keys (any subset is fine):
          - ts: float (required)
          - robot_position: np.ndarray[float] (J,) - current joint positions
          - robot_target_cartesian: np.ndarray[float] (9,) - right, left, torso Cartesian positions
          - robot_target_joints: np.ndarray[float] (J,) - target joint angles from IK (all joints)
          - torso_target_joints: np.ndarray[float] (J,) - torso target (indices 0-5), rest NaN
          - right_arm_target_joints: np.ndarray[float] (J,) - right arm target (indices 6-12), rest NaN
          - left_arm_target_joints: np.ndarray[float] (J,) - left arm target (indices 13-19), rest NaN
          - gripper_state: np.ndarray[float] (G,)
          - gripper_target: np.ndarray[float] (G,)
          - base_state: np.ndarray[float] (3,) - linear_x, linear_y, angular_z
          - head_rgb: np.ndarray[uint8] shape (H, W, 3)
          - head_rgb_ts: float  # optional; defaults to ts
          - head_depth: np.ndarray[uint16] shape (H, W)
          - head_depth_ts: float  # optional; defaults to ts
          - pcd_points: np.ndarray[float32] shape (N, 3) - point cloud XYZ coordinates
          - pcd_colors: np.ndarray[uint8] shape (N, 3) - point cloud RGB colors
        """
        try:
            self.q.put_nowait(sample)
        except queue.Full:
            if drop_oldest:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(sample)
                except queue.Full:
                    pass
            else:
                self.q.put(sample)

    def _run(self):
        f = None
        # kinematics / gripper datasets
        d_time = d_robot = d_robot_target_cart = d_robot_target_joints = None
        d_torso_target = d_right_arm_target = d_left_arm_target = None
        d_grip = d_grip_target = d_base = None
        idx = 0
        # 현재까지 저장된 샘플 수를 추적 (업데이트 시 사용)
        saved_sample_count = 0

        # camera datasets for all groups (head, right_wrist, left_wrist)
        camera_groups = ['head', 'right']
        d_cam_data = {}  # {'head': {'time': dataset, 'img': dataset, 'idx': 0}, ...}
        for cam in camera_groups:
            d_cam_data[cam] = {'time': None, 'img': None, 'idx': 0, 'depth_time': None, 'depth_img': None, 'depth_idx': 0}

        # PCD datasets
        d_pcd_time = d_pcd_points = d_pcd_colors = None
        pcd_idx = 0

        last_flush = time.perf_counter()

        try:
            f = h5py.File(self.path, "w")
            grp = f.create_group("samples")
            # camera stored in separate groups for each camera
            camera_grps = {}
            for cam in camera_groups:
                camera_grps[cam] = {
                    'rgb': f.create_group(f"{cam}_rgb"),
                    'depth': f.create_group(f"{cam}_depth")
                }

            # PCD group
            pcd_grp = f.create_group("pointclouds") if self.save_pcd else None
            batch = []

            while not self.stop_event.is_set():
                try:
                    item = self.q.get(timeout=0.2)
                except queue.Empty:
                    item = None

                if item is not None:
                    batch.append(item)

                # ----- Lazy dataset creation for kinematics/gripper -----
                if (d_time is None) and batch:
                    first = next((s for s in batch if s.get("robot_position") is not None), None)
                    rp_dim = len(first["robot_position"]) if first else 0

                    # first_rtc = next((s for s in batch if s.get("robot_target_cartesian") is not None), None)
                    # rtc_dim = len(first_rtc["robot_target_cartesian"]) if first_rtc else 0

                    # All joint target arrays have same dimension as robot_position
                    joint_dim = rp_dim

                    first_g = next((s for s in batch if s.get("gripper_state") is not None), None)
                    gs_dim = len(first_g["gripper_state"]) if first_g else 0

                    first_b = next((s for s in batch if s.get("base_state") is not None), None)
                    bs_dim = len(first_b["base_state"]) if first_b else 0

                    d_time = grp.create_dataset("time", shape=(0,), maxshape=(None,),
                                                dtype="f8", chunks=True, compression="gzip")
                    d_robot = (grp.create_dataset("robot_position",
                                                  shape=(0, rp_dim), maxshape=(None, rp_dim),
                                                  dtype="f8", chunks=True, compression="gzip")
                               if rp_dim > 0 else None)
                    # d_robot_target_cart = (grp.create_dataset("robot_target_cartesian",
                    #                               shape=(0, rtc_dim), maxshape=(None, rtc_dim),
                    #                               dtype="f8", chunks=True, compression="gzip")
                    #            if rtc_dim > 0 else None)
                    d_robot_target_joints = (grp.create_dataset("robot_target_joints",
                                                  shape=(0, joint_dim), maxshape=(None, joint_dim),
                                                  dtype="f8", chunks=True, compression="gzip")
                               if joint_dim > 0 else None)
                    d_grip = (grp.create_dataset("gripper_state",
                                                 shape=(0, gs_dim), maxshape=(None, gs_dim),
                                                 dtype="f8", chunks=True, compression="gzip")
                              if gs_dim > 0 else None)
                    d_grip_target = (grp.create_dataset("gripper_target",
                                                        shape=(0, gs_dim), maxshape=(None, gs_dim),
                                                        dtype="f8", chunks=True, compression="gzip")
                                     if gs_dim > 0 else None)
                    d_base = (grp.create_dataset("base_state",
                                                 shape=(0, bs_dim), maxshape=(None, bs_dim),
                                                 dtype="f8", chunks=True, compression="gzip")
                              if bs_dim > 0 else None)

                # ----- Lazy dataset creation for all camera groups -----
                for cam in camera_groups:
                    if (d_cam_data[cam]['img'] is None) and batch:
                        key = f"{cam}_rgb"
                        cam_first = next((s for s in batch if s.get(key) is not None), None)
                        if cam_first is not None:
                            try:
                                img = np.asarray(cam_first[key])
                                if img.ndim != 3 or img.shape[2] not in (3, 4):
                                    raise ValueError(f"{key} must be HxWx3(4) uint8, got {img.shape}")
                                H, W, C = img.shape
                                logging.info(f"[h5_writer] Creating {key} dataset: shape=({H}, {W}, {C})")
                                # timestamps for camera frames
                                d_cam_data[cam]['time'] = camera_grps[cam]['rgb'].create_dataset(
                                    "time", shape=(0,), maxshape=(None,), dtype="f8",
                                    chunks=True, compression="gzip"
                                )
                                # frames (uint8), chunk on single frame for fast appends
                                d_cam_data[cam]['img'] = camera_grps[cam]['rgb'].create_dataset(
                                    "image", shape=(0, H, W, C), maxshape=(None, H, W, C),
                                    dtype="u1", chunks=(1, H, W, C), compression="gzip"
                                )
                            except Exception as e:
                                logging.error(f"[h5_writer] Failed to create {key} dataset: {e}")
                                continue

                # ----- Lazy dataset creation for all depth groups -----
                for cam in camera_groups:
                    if (d_cam_data[cam]['depth_img'] is None) and batch:
                        key = f"{cam}_depth"
                        depth_first = next((s for s in batch if s.get(key) is not None), None)
                        if depth_first is not None:
                            try:
                                depth = np.asarray(depth_first[key])
                                if depth.ndim != 2:
                                    raise ValueError(f"{key} must be HxW uint16, got {depth.shape}")
                                H_d, W_d = depth.shape
                                logging.info(f"[h5_writer] Creating {key} dataset: shape=({H_d}, {W_d})")
                                # timestamps for depth frames
                                d_cam_data[cam]['depth_time'] = camera_grps[cam]['depth'].create_dataset(
                                    "time", shape=(0,), maxshape=(None,), dtype="f8",
                                    chunks=True, compression="gzip"
                                )
                                # depth frames (uint16), chunk on single frame
                                d_cam_data[cam]['depth_img'] = camera_grps[cam]['depth'].create_dataset(
                                    "image", shape=(0, H_d, W_d), maxshape=(None, H_d, W_d),
                                    dtype="u2", chunks=(1, H_d, W_d), compression="gzip"
                                )
                            except Exception as e:
                                logging.error(f"[h5_writer] Failed to create {key} dataset: {e}")
                                continue

                # ----- Append kinematics / gripper -----
                if batch and d_time is not None:
                    ts_list, rp_list, rtc_list, rtj_list = [], [], [], []
                    torso_list, rarm_list, larm_list = [], [], []
                    gs_list, gs_target_list, bs_list = [], [], []

                    for s in batch:
                        ts_list.append(float(s["ts"]))
                        if d_robot is not None:
                            rp_list.append(
                                [np.nan] * d_robot.shape[1]
                                if s.get("robot_position") is None else s["robot_position"]
                            )
                        # if d_robot_target_cart is not None:
                        #     rtc_list.append(
                        #         [np.nan] * d_robot_target_cart.shape[1]
                        #         if s.get("robot_target_cartesian") is None else s["robot_target_cartesian"]
                        #     )
                        if d_robot_target_joints is not None:
                            rtj_list.append(
                                [np.nan] * d_robot_target_joints.shape[1]
                                if s.get("robot_target_joints") is None else s["robot_target_joints"]
                            )
                        if d_grip is not None:
                            gs_list.append(
                                [np.nan] * d_grip.shape[1]
                                if s.get("gripper_state") is None else s["gripper_state"]
                            )
                        if d_grip_target is not None:
                            # if missing, fallback to gripper_state
                            if s.get("gripper_target") is None and s.get("gripper_state") is not None:
                                s["gripper_target"] = s["gripper_state"]
                            gs_target_list.append(
                                [np.nan] * d_grip_target.shape[1]
                                if s.get("gripper_target") is None else s["gripper_target"]
                            )
                        if d_base is not None:
                            bs_list.append(
                                [np.nan] * d_base.shape[1]
                                if s.get("base_state") is None else s["base_state"]
                            )

                    n_new = len(ts_list)
                    d_time.resize((idx + n_new,))
                    d_time[idx: idx + n_new] = np.asarray(ts_list, dtype=np.float64)

                    if d_robot is not None:
                        d_robot.resize((idx + n_new, d_robot.shape[1]))
                        d_robot[idx: idx + n_new, :] = np.asarray(rp_list, dtype=np.float64)

                    if d_robot_target_cart is not None:
                        d_robot_target_cart.resize((idx + n_new, d_robot_target_cart.shape[1]))
                        d_robot_target_cart[idx: idx + n_new, :] = np.asarray(rtc_list, dtype=np.float64)

                    if d_robot_target_joints is not None:
                        d_robot_target_joints.resize((idx + n_new, d_robot_target_joints.shape[1]))
                        d_robot_target_joints[idx: idx + n_new, :] = np.asarray(rtj_list, dtype=np.float64)
                    
                    if d_torso_target is not None:
                        d_torso_target.resize((idx + n_new, d_torso_target.shape[1]))
                        d_torso_target[idx: idx + n_new, :] = np.asarray(torso_list, dtype=np.float64)

                    if d_right_arm_target is not None:
                        d_right_arm_target.resize((idx + n_new, d_right_arm_target.shape[1]))
                        d_right_arm_target[idx: idx + n_new, :] = np.asarray(rarm_list, dtype=np.float64)

                    if d_left_arm_target is not None:
                        d_left_arm_target.resize((idx + n_new, d_left_arm_target.shape[1]))
                        d_left_arm_target[idx: idx + n_new, :] = np.asarray(larm_list, dtype=np.float64)

                    if d_grip is not None:
                        d_grip.resize((idx + n_new, d_grip.shape[1]))
                        d_grip[idx: idx + n_new, :] = np.asarray(gs_list, dtype=np.float64)

                    if d_grip_target is not None:
                        d_grip_target.resize((idx + n_new, d_grip_target.shape[1]))
                        d_grip_target[idx: idx + n_new, :] = np.asarray(gs_target_list, dtype=np.float64)

                    if d_base is not None:
                        d_base.resize((idx + n_new, d_base.shape[1]))
                        d_base[idx: idx + n_new, :] = np.asarray(bs_list, dtype=np.float64)

                    # ----- Update previous step's robot_target_joints with current robot_pos -----
                    # batch 처리와 관계없이 queue에 있는 업데이트 요청을 하나씩 처리
                    try:
                        while not self._update_queue.empty():
                            target_idx, robot_pos = self._update_queue.get_nowait()
                            if d_robot_target_joints is not None and robot_pos is not None:
                                # target_idx에 해당하는 robot_target_joints를 robot_pos로 업데이트
                                if target_idx >= 0 and target_idx < d_robot_target_joints.shape[0]:
                                    d_robot_target_joints[target_idx, :] = np.asarray(robot_pos, dtype=np.float64)
                                else:
                                    logging.warning(f"[h5_writer] Invalid target_idx {target_idx} (shape: {d_robot_target_joints.shape[0]})")
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logging.warning(f"[h5_writer] Failed to update previous target: {e}")
                    
                    # 저장된 샘플 수 업데이트
                    idx += n_new
                    self._current_idx = idx

                # ----- Append camera frames (independent index) -----
                for cam in camera_groups:
                    if batch and (d_cam_data[cam]['img'] is not None):
                        cam_times, cam_imgs = [], []
                        key_rgb = f"{cam}_rgb"
                        key_rgb_ts = f"{cam}_rgb_ts"
                        for s in batch:
                            if s.get(key_rgb) is not None:
                                try:
                                    img = np.asarray(s[key_rgb])
                                    # (optional) validate shape matches dataset
                                    if img.shape != d_cam_data[cam]['img'].shape[1:]:
                                        logging.warning(
                                            f"[h5_writer] {key_rgb} shape {img.shape} does not match dataset {d_cam_data[cam]['img'].shape[1:]} - skipping"
                                        )
                                        continue
                                    cam_imgs.append(img.astype(np.uint8, copy=False))
                                    cam_times.append(float(s.get(key_rgb_ts, s["ts"])))
                                except Exception as e:
                                    logging.warning(f"[h5_writer] Failed to process {key_rgb}: {e}")
                                    continue

                        if cam_times:
                            n_new_cam = len(cam_times)
                            cam_idx = d_cam_data[cam]['idx']
                            d_cam_data[cam]['time'].resize((cam_idx + n_new_cam,))
                            d_cam_data[cam]['time'][cam_idx: cam_idx + n_new_cam] = np.asarray(cam_times, dtype=np.float64)

                            d_cam_data[cam]['img'].resize((cam_idx + n_new_cam, *d_cam_data[cam]['img'].shape[1:]))
                            d_cam_data[cam]['img'][cam_idx: cam_idx + n_new_cam, ...] = np.stack(cam_imgs, axis=0)

                            d_cam_data[cam]['idx'] += n_new_cam

                # ----- Append depth frames (independent index) -----
                for cam in camera_groups:
                    if batch and (d_cam_data[cam]['depth_img'] is not None):
                        depth_times, depth_imgs = [], []
                        key_depth = f"{cam}_depth"
                        key_depth_ts = f"{cam}_depth_ts"
                        
                        for s in batch:
                            if s.get(key_depth) is not None:
                                try:
                                    depth = np.asarray(s[key_depth])
                                    # validate shape matches dataset
                                    if depth.shape != d_cam_data[cam]['depth_img'].shape[1:]:
                                        logging.warning(
                                            f"[h5_writer] {key_depth} shape {depth.shape} does not match dataset {d_cam_data[cam]['depth_img'].shape[1:]} - skipping"
                                        )
                                        continue
                                    depth_imgs.append(depth.astype(np.uint16, copy=False))
                                    depth_times.append(float(s.get(key_depth_ts, s["ts"])))
                                except Exception as e:
                                    logging.warning(f"[h5_writer] Failed to process {key_depth}: {e}")
                                    continue

                        if depth_times:
                            n_new_depth = len(depth_times)
                            depth_idx = d_cam_data[cam]['depth_idx']
                            d_cam_data[cam]['depth_time'].resize((depth_idx + n_new_depth,))
                            d_cam_data[cam]['depth_time'][depth_idx: depth_idx + n_new_depth] = np.asarray(depth_times, dtype=np.float64)

                            d_cam_data[cam]['depth_img'].resize((depth_idx + n_new_depth, *d_cam_data[cam]['depth_img'].shape[1:]))
                            d_cam_data[cam]['depth_img'][depth_idx: depth_idx + n_new_depth, ...] = np.stack(depth_imgs, axis=0)

                            d_cam_data[cam]['depth_idx'] += n_new_depth
                
                # ----- Append PCD data (from pre-computed point clouds) -----
                if batch and self.save_pcd and pcd_grp is not None:
                    pcd_times_list = []
                    pcd_points_list = []
                    pcd_colors_list = []
                    
                    for s in batch:
                        if s.get("pcd_points") is not None and s.get("pcd_colors") is not None:
                            points = np.asarray(s["pcd_points"])
                            colors = np.asarray(s["pcd_colors"])
                            ts = float(s.get("head_depth_ts", s.get("head_rgb_ts", s["ts"])))
                            
                            # Lazy create PCD datasets
                            if d_pcd_points is None:
                                # Create variable length datasets for PCD
                                dt_points = h5py.vlen_dtype(np.dtype('float32'))
                                dt_colors = h5py.vlen_dtype(np.dtype('uint8'))
                                
                                d_pcd_time = pcd_grp.create_dataset(
                                    "time", shape=(0,), maxshape=(None,), dtype="f8",
                                    chunks=True, compression="gzip"
                                )
                                d_pcd_points = pcd_grp.create_dataset(
                                    "points", shape=(0,), maxshape=(None,), dtype=dt_points,
                                    chunks=True, compression="gzip"
                                )
                                d_pcd_colors = pcd_grp.create_dataset(
                                    "colors", shape=(0,), maxshape=(None,), dtype=dt_colors,
                                    chunks=True, compression="gzip"
                                )
                            
                            # Flatten for variable-length storage
                            pcd_times_list.append(ts)
                            pcd_points_list.append(points.flatten().astype(np.float32))
                            pcd_colors_list.append(colors.flatten().astype(np.uint8))
                    
                    # Append to HDF5
                    if pcd_times_list:
                        n_new_pcd = len(pcd_times_list)
                        d_pcd_time.resize((pcd_idx + n_new_pcd,))
                        d_pcd_time[pcd_idx: pcd_idx + n_new_pcd] = np.asarray(pcd_times_list, dtype=np.float64)
                        
                        d_pcd_points.resize((pcd_idx + n_new_pcd,))
                        d_pcd_points[pcd_idx: pcd_idx + n_new_pcd] = pcd_points_list
                        
                        d_pcd_colors.resize((pcd_idx + n_new_pcd,))
                        d_pcd_colors[pcd_idx: pcd_idx + n_new_pcd] = pcd_colors_list
                        
                        pcd_idx += n_new_pcd

                # done with this batch
                if batch:
                    batch.clear()

                # flushing policy
                now = time.perf_counter()
                should_flush = ((idx % self.flush_every == 0 and idx > 0) or
                               (now - last_flush > self.flush_secs))
                
                # Check all camera groups for flushing
                for cam in camera_groups:
                    if d_cam_data[cam]['idx'] % self.flush_every == 0 and d_cam_data[cam]['idx'] > 0:
                        should_flush = True
                    if d_cam_data[cam]['depth_idx'] % self.flush_every == 0 and d_cam_data[cam]['depth_idx'] > 0:
                        should_flush = True
                
                # Check PCD
                should_flush = should_flush or (pcd_idx % self.flush_every == 0 and pcd_idx > 0)
                
                if should_flush:
                    f.flush()
                    last_flush = now

                if item is None and self.stop_event.is_set() and self.q.empty():
                    break

        finally:
            if f is not None:
                f.flush()
                f.close()
