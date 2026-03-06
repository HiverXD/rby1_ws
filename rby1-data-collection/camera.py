import gc
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

        # Discover available devices (temporary context — released before pipeline creation)
        _discovery_ctx = rs.context()
        available_serials = {
            dev.get_info(rs.camera_info.serial_number)
            for dev in _discovery_ctx.query_devices()
        }
        del _discovery_ctx
        gc.collect()
        logging.info(f"연결된 RealSense 장치 ({len(available_serials)}개): {sorted(available_serials)}")

        # Per-camera pipeline creation (each pipeline gets its own internal context)
        for serial in self.serials:
            if serial in available_serials:
                self._make_pipeline_and_config(serial)
                self.buffers[serial] = deque(maxlen=self.buffer_size)
                logging.info(f"Camera {serial} configured.")
            else:
                logging.warning(
                    f"Camera {serial} NOT FOUND — USB 연결 확인 필요. "
                    f"감지된 장치: {sorted(available_serials)}"
                )

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

    def _make_pipeline_and_config(self, serial: str):
        """새 pipeline/config/align 을 생성하여 해당 시리얼에 등록합니다.
        
        rs.pipeline()을 context 인자 없이 생성하면 내부적으로 독립 context를
        만들므로, 카메라 간 USB 핸들이 격리되어 한 카메라의 실패가 다른
        카메라에 영향을 주지 않습니다.
        """
        pipe = rs.pipeline()  # 독립 context 자동 생성 → 카메라 간 격리
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipelines[serial] = pipe
        self.configs[serial] = cfg
        self.aligns[serial] = rs.align(rs.stream.color)

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
                    time.sleep(1.0)  # USB 대역폭 안정화 — 다음 카메라 시작 전 대기
                    break
                except Exception as e:
                    retry_count += 1
                    err_msg = str(e)

                    # ── 실패 직후 장치 핸들 해제 ───────────────────────
                    # pipeline.start()가 중간에 실패하면 UVC 장치가 half-open 상태로
                    # 남아 재시도 시 "already opened" 오류가 발생합니다.
                    # pipeline.stop()으로 핸들을 먼저 해제한 뒤 새 객체를 생성합니다.
                    try:
                        self.pipelines[serial].stop()
                    except Exception:
                        pass
                    time.sleep(2.0)

                    if retry_count < max_retries:
                        logging.warning(
                            f"Failed to start pipeline for camera {serial} "
                            f"(attempt {retry_count}/{max_retries}): {err_msg}"
                        )
                        # 마지막 재시도 전: 하드웨어 리셋으로 UVC 상태 초기화
                        if retry_count == max_retries - 1:
                            try:
                                ctx_tmp = rs.context()
                                for dev in ctx_tmp.query_devices():
                                    if dev.get_info(rs.camera_info.serial_number) == serial:
                                        logging.warning(f"[{serial}] 하드웨어 리셋 시도...")
                                        dev.hardware_reset()
                                        break
                                del ctx_tmp
                                gc.collect()
                                time.sleep(5.0)  # 리셋 후 USB 재열거 대기
                            except Exception as hw_e:
                                logging.warning(f"[{serial}] 하드웨어 리셋 실패 (무시): {hw_e}")
                        # 새 pipeline/config 객체 생성 (half-open 객체 버림)
                        self._make_pipeline_and_config(serial)
                        gc.collect()
                    else:
                        logging.error(
                            f"Failed to start pipeline for camera {serial} "
                            f"after {max_retries} attempts: {err_msg} (type={type(e).__name__})"
                        )
        
        if not started_serials:
            logging.error("No cameras could be started.")
            return

        # 시작 실패한 카메라는 buffers에서 제거 (sync 루프가 영구 blocked 되는 버그 방지)
        failed_serials = [s for s in list(self.buffers.keys()) if s not in started_serials]
        for serial in failed_serials:
            logging.warning(f"Camera {serial} failed to start — removing from sync loop")
            self.buffers.pop(serial, None)

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

        try:
            self.pipelines[serial].stop()
        except Exception:
            pass

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            # 매 시도마다 새 pipeline/config 생성 (half-open 상태 방지)
            self._make_pipeline_and_config(serial)
            gc.collect()
            time.sleep(0.5)
            try:
                profile = self.pipelines[serial].start(self.configs[serial])
                self._apply_auto_controls(profile, serial)
                with self.lock:
                    if serial in self.buffers:
                        self.buffers[serial].clear()
                    self.synced_frames = {}
                logging.info(f"[{serial}] pipeline restart succeeded (attempt {attempt}/{max_retries})")
                return True
            except Exception as e:
                logging.warning(f"[{serial}] pipeline restart failed (attempt {attempt}/{max_retries}): {e}")
                try:
                    self.pipelines[serial].stop()
                except Exception:
                    pass
                time.sleep(1.0)

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

                depth_image = np.array(depth_frame.get_data(), copy=True)
                color_image = np.array(color_frame.get_data(), copy=True)

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
        모든 카메라 버퍼에서 가장 최신 프레임을 골라 프레임 세트로 만든다.

        중요: 하드웨어 동기화 없는 독립 카메라에서는 센서 타임스탬프가
        각각 다른 클록 도메인(hardware_clock)일 수 있으므로, tolerance를 초과해도
        sync를 항상 성공시키고 각 카메라의 최신 프레임을 반환한다.
        """
        interval = max(0.001, 1.0 / self.fps / 2)

        ref_serial = next(iter(self.buffers), None)
        if ref_serial is None:
            logging.error("No cameras started — sync loop exiting.")
            return
        logging.info(f"Sync reference camera: {ref_serial}")

        while self.running:
            with self.lock:
                ref_buf = self.buffers.get(ref_serial)
                if not ref_buf:
                    pass
                else:
                    ref_frame = ref_buf[-1]
                    candidates: Dict[str, RGBDFrame] = {ref_serial: ref_frame}
                    all_have_data = True

                    for serial, buf in self.buffers.items():
                        if serial == ref_serial:
                            continue
                        if len(buf) == 0:
                            all_have_data = False
                            break
                        # 항상 최신 프레임 사용 (타임스탬프 도메인 차이에 무관)
                        candidates[serial] = buf[-1]

                    if all_have_data:
                        self.synced_frames = candidates

                        # 메모리 관리: 각 버퍼에 최근 5개만 유지
                        for serial, buf in self.buffers.items():
                            while len(buf) > 5:
                                buf.popleft()

            time.sleep(interval)

    def get_frames(self) -> Dict[str, RGBDFrame]:
        """
        동기화된 프레임 세트를 반환.
        _capture_loop에서 이미 np.array(copy=True)로 복사하므로
        deep copy 없이 dict 앉은 복사만 수행합니다 (46MB/sec → ~0 절감).
        """
        with self.lock:
            return dict(self.synced_frames)  # shallow dict copy — O(N cameras)

    def stop(self):
        self.running = False

        # Join threads
        for th in self.capture_threads.values():
            try:
                th.join(timeout=2.0)
            except Exception:
                pass

        if self.sync_thread:
            try:
                self.sync_thread.join(timeout=2.0)
            except Exception:
                pass

        # Stop pipelines
        for serial, pipe in self.pipelines.items():
            try:
                pipe.stop()
                logging.info(f"Pipeline stopped for camera {serial}")
            except Exception as e:
                logging.error(f"Failed to stop pipeline for camera {serial}: {e}")

        # 모든 참조 해제 → USB 핸들 완전 반환
        self.pipelines.clear()
        self.configs.clear()
        self.aligns.clear()
        self.filters.clear()
        self.capture_threads.clear()
        self.sync_thread = None
        with self.lock:
            self.buffers.clear()
            self.synced_frames.clear()
        gc.collect()
