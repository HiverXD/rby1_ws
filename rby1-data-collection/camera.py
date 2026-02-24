import copy
import threading
import logging
import numpy as np
import time
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque


@dataclass
class RGBDFrame:
    # 센서 timestamp(초 단위)로 저장
    t: float
    color: np.ndarray
    depth: np.ndarray
    serial: str


class MultiRealsense:
    def __init__(
        self,
        camera_serials: List[str],
        width=640,
        height=480,
        fps=30,
        sync_tolerance_ms: float = 30.0,
        buffer_size: int = 30,
    ):
        self.serials = camera_serials
        self.width = width
        self.height = height
        self.fps = fps

        # 동기화 설정
        self.sync_tolerance_ms = float(sync_tolerance_ms)
        self.buffer_size = int(buffer_size)

        self.pipelines: Dict[str, rs.pipeline] = {}
        self.configs: Dict[str, rs.config] = {}
        self.aligns: Dict[str, rs.align] = {}

        self.filters: Dict[str, Dict] = {} #key: Serial Number

        # Timeout/recovery settings for long-running capture sessions
        self.frame_wait_timeout_ms: int = 5000
        self.max_consecutive_timeouts: int = 3
        self.restart_cooldown_sec: float = 2.0

        self.running = False

        # 카메라별 캡처 스레드 + 동기화 스레드
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.sync_thread: Optional[threading.Thread] = None

        # 카메라별 프레임 버퍼 (timestamp 순)
        self.buffers: Dict[str, deque] = {}
        self.lock = threading.Lock()

        # 최종 “동기화된” 프레임 세트
        self.synced_frames: Dict[str, RGBDFrame] = {}

        # Discover and initialize cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        available_serials = {dev.get_info(rs.camera_info.serial_number) for dev in devices}

        for serial in self.serials:
            if serial in available_serials:
                pipe = rs.pipeline(ctx)
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

                self.pipelines[serial] = pipe
                self.configs[serial] = config
                self.aligns[serial] = rs.align(rs.stream.color)

                self.buffers[serial] = deque(maxlen=self.buffer_size)
                logging.info(f"Camera {serial} configured.")
            else:
                logging.warning(f"Camera with serial {serial} not found.")

    def enable_filters_for_serial(self, serial: str):
        """
        해당 시리얼 번호를 가진 카메라에 대해 Post-Processing 필터를 생성하고 설정합니다.
        """
        if serial not in self.pipelines:
            logging.warning(f"Cannot enable filters: Camera {serial} not initialized.")
            return

        logging.info(f"✨ Enabling Depth Filters for Camera {serial}")
        
        # 필터 객체 생성 및 옵션 설정
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 1) # 해상도 유지 (필요시 조절)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4) # 떨림 vs 잔상 타협점
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        # temporal.set_option(rs.option.holes_fill, 3) # Temporal 자체 persistence 사용 가능

        # 딕셔너리에 저장
        self.filters[serial] = {
            'decimation': decimation,
            'temporal': temporal
        }

    def start(self):
        if not self.pipelines:
            logging.error("No cameras configured. Cannot start.")
            return

        # Track which pipelines started successfully
        started_serials = []

        # Start pipelines with retry logic
        for serial in list(self.pipelines.keys()):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    profile = self.pipelines[serial].start(self.configs[serial])
                    self._apply_auto_controls(profile, serial)

                    logging.info(f"Pipeline started for camera {serial}")
                    started_serials.append(serial)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"Failed to start pipeline for camera {serial} (attempt {retry_count}/{max_retries}): {e}")
                        time.sleep(2)  # Wait before retry
                    else:
                        logging.error(f"Failed to start pipeline for camera {serial} after {max_retries} attempts: {e}")
        
        if not started_serials:
            logging.error("No cameras could be started.")
            return

        self.running = True

        # Start per-camera capture threads (only for successfully started cameras)
        for serial in started_serials:
            th = threading.Thread(target=self._capture_loop, args=(serial,), daemon=True)
            self.capture_threads[serial] = th
            th.start()

        # Start sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

    def _apply_auto_controls(self, profile, serial: str):
        # Ensure color auto controls are enabled for stable RGB color.
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                if sensor.supports(rs.option.enable_auto_white_balance):
                    sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception as e:
            logging.warning(f"[{serial}] failed to set auto exposure/white balance: {e}")

    def _restart_pipeline(self, serial: str) -> bool:
        if serial not in self.pipelines:
            return False

        pipe = self.pipelines[serial]
        cfg = self.configs[serial]

        try:
            pipe.stop()
        except Exception:
            pass

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                profile = pipe.start(cfg)
                self._apply_auto_controls(profile, serial)
                with self.lock:
                    if serial in self.buffers:
                        self.buffers[serial].clear()
                    self.synced_frames = {}
                logging.info(f"[{serial}] pipeline restart succeeded (attempt {attempt}/{max_retries})")
                return True
            except Exception as e:
                logging.warning(f"[{serial}] pipeline restart failed (attempt {attempt}/{max_retries}): {e}")
                time.sleep(0.5)

        logging.error(f"[{serial}] pipeline restart failed after {max_retries} attempts")
        return False

    def _capture_loop(self, serial: str):
        """
        카메라 1대당 1스레드로 프레임을 지속 수집해 버퍼에 적재.
        """
        pipe = self.pipelines[serial]
        align = self.aligns[serial]
        consecutive_timeouts = 0
        last_timeout_log_t = 0.0
        last_restart_t = 0.0

        while self.running:
            try:
                frameset = pipe.wait_for_frames(timeout_ms=self.frame_wait_timeout_ms)
                aligned = align.process(frameset)

                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    logging.warning(f"[{serial}] capture failed: depth or color frame is None")
                    continue

                # logging.info(f"[{serial}] capture successful.")

                # [추가] 필터 적용 로직
                # 해당 시리얼 번호에 필터가 등록되어 있다면 적용
                if serial in self.filters:
                    filters = self.filters[serial]
                    
                    # 필터 체인 적용 (Spatial -> Temporal -> Hole Filling)
                    # Decimation은 Align 후에 적용하면 해상도가 틀어질 수 있어 주의 필요 (여기선 제외하거나 맨 앞에 적용)
                    # depth_frame = filters['decimation'].process(depth_frame) 
                    
                    depth_frame = filters['temporal'].process(depth_frame)

                # ✅ 센서 timestamp 사용 (ms) -> seconds
                # color 기준으로 timestamp 사용 (depth도 같은 frameset 기반이라 근접)
                ts_ms = float(color_frame.get_timestamp())
                ts_s = ts_ms / 1000.0

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                fr = RGBDFrame(
                    t=ts_s,
                    color=color_image,
                    depth=depth_image,
                    serial=serial
                )

                with self.lock:
                    self.buffers[serial].append(fr)

                # frame received successfully: reset timeout counter
                consecutive_timeouts = 0

            except Exception as e:
                msg = str(e).lower()
                is_timeout = (
                    ("timeout" in msg)
                    or ("timed out" in msg)
                    or ("frame didn't arrive" in msg)
                )

                if is_timeout:
                    consecutive_timeouts += 1
                    now = time.perf_counter()

                    if now - last_timeout_log_t > 1.0:
                        logging.warning(
                            f"[{serial}] frame timeout ({consecutive_timeouts}/{self.max_consecutive_timeouts})"
                        )
                        last_timeout_log_t = now

                    if (
                        consecutive_timeouts >= self.max_consecutive_timeouts
                        and (now - last_restart_t) >= self.restart_cooldown_sec
                    ):
                        logging.error(f"[{serial}] too many frame timeouts, restarting pipeline...")
                        self._restart_pipeline(serial)
                        # refresh local references after restart
                        pipe = self.pipelines[serial]
                        align = self.aligns[serial]
                        last_restart_t = now
                        consecutive_timeouts = 0

                    continue

                logging.warning(f"[{serial}] capture failed: {e}")
                time.sleep(0.05)
                continue

    def _sync_loop(self):
        """
        모든 카메라 버퍼에서 timestamp가 가장 비슷한 프레임을 골라
        하나의 “동기화된 프레임 세트”로 만든다.
        """
        tol_s = self.sync_tolerance_ms / 1000.0

        # 기준 카메라(첫 번째)를 레퍼런스로 사용
        ref_serial = self.serials[0] if self.serials else None
        if ref_serial is None or ref_serial not in self.buffers:
            logging.error("No reference camera available for sync.")
            return

        while self.running:
            with self.lock:
                # 레퍼런스 버퍼가 비어 있으면 기다림
                if len(self.buffers[ref_serial]) == 0:
                    pass
                else:
                    # 레퍼런스의 “가장 최신” 프레임을 기준시각으로 사용
                    ref_frame = self.buffers[ref_serial][-1]
                    t_ref = ref_frame.t

                    candidate_set: Dict[str, RGBDFrame] = {ref_serial: ref_frame}
                    ok = True

                    # 다른 카메라에서 t_ref에 가장 가까운 프레임 찾기
                    for serial, buf in self.buffers.items():
                        if serial == ref_serial:
                            continue
                        if len(buf) == 0:
                            ok = False
                            break

                        # buf 안에서 |t - t_ref| 최소인 프레임 선택
                        best = min(buf, key=lambda f: abs(f.t - t_ref))
                        if abs(best.t - t_ref) > tol_s:
                            ok = False
                            break
                        candidate_set[serial] = best

                    if ok:
                        # ✅ 동기화 성공: synced_frames 갱신
                        self.synced_frames = copy.deepcopy(candidate_set)

                        # (선택) 너무 오래된 프레임 정리:
                        # 기준시각보다 한참 과거 프레임들은 버퍼에서 제거해 지연/메모리 감소
                        for serial, buf in self.buffers.items():
                            # t_ref - 2*tol 보다 오래된 건 제거
                            while len(buf) > 0 and buf[0].t < (t_ref - 2 * tol_s):
                                buf.popleft()

            # sync 루프 주기 (너무 빠르게 돌지 않게)
            time.sleep(max(0, 1.0 / self.fps / 2))

    def get_frames(self) -> Dict[str, RGBDFrame]:
        """
        ✅ 동기화된 프레임 세트를 반환.
        모든 시리얼이 존재하지 않을 수도 있으니(초기 구간/카메라 드랍) 호출부에서 체크 권장.
        """
        with self.lock:
            return copy.deepcopy(self.synced_frames)

    def stop(self):
        self.running = False

        # Join threads
        for th in self.capture_threads.values():
            try:
                th.join(timeout=1.0)
            except Exception:
                pass

        if self.sync_thread:
            try:
                self.sync_thread.join(timeout=1.0)
            except Exception:
                pass

        # Stop pipelines
        for serial, pipe in self.pipelines.items():
            try:
                pipe.stop()
                logging.info(f"Pipeline stopped for camera {serial}")
            except Exception as e:
                logging.error(f"Failed to stop pipeline for camera {serial}: {e}")

        self.capture_threads = {}
        self.sync_thread = None
        with self.lock:
            self.synced_frames = {}
