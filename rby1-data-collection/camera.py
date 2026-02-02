import copy
import threading
import logging
import numpy as np
import time
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RGBDFrame:
    t: float
    color: np.ndarray
    depth: np.ndarray
    serial: str


class MultiRealsense:
    def __init__(self, camera_serials: List[str], width=640, height=480, fps=30):
        self.serials = camera_serials
        self.width = width
        self.height = height
        self.fps = fps
        self.pipelines = {}
        self.aligns = {}
        self.running = False
        self.thread = None

        self.frames: Dict[str, RGBDFrame] = {}
        self.lock = threading.Lock()

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
                self.pipelines[serial] = (pipe, config)
                self.aligns[serial] = rs.align(rs.stream.color)
                logging.info(f"Camera {serial} configured.")
            else:
                logging.warning(f"Camera with serial {serial} not found.")

    def start(self):
        if not self.pipelines:
            logging.error("No cameras configured. Cannot start.")
            return

        for serial, (pipe, config) in self.pipelines.items():
            try:
                pipe.start(config)
                logging.info(f"Pipeline started for camera {serial}")
            except Exception as e:
                logging.error(f"Failed to start pipeline for camera {serial}: {e}")
                # clean up already started pipelines
                self.stop()
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            framesets = {}
            for serial, (pipe, _) in self.pipelines.items():
                try:
                    frames = pipe.wait_for_frames(timeout_ms=1000)
                    framesets[serial] = frames
                except Exception as e:
                    logging.warning(f"Did not get frame from {serial}: {e}")
                    # If a camera fails, maybe we should remove it from the list for a while
                    continue
            
            with self.lock:
                self.frames.clear()
                for serial, frameset in framesets.items():
                    aligned_frames = self.aligns[serial].process(frameset)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        continue
                    
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    self.frames[serial] = RGBDFrame(
                        t=time.time(),
                        color=color_image,
                        depth=depth_image,
                        serial=serial
                    )
            # Adjust sleep time to be more precise
            time.sleep(max(0, 1.0/self.fps - 0.005))

    def get_frames(self) -> Dict[str, RGBDFrame]:
        with self.lock:
            return copy.deepcopy(self.frames)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        for serial, (pipe, _) in self.pipelines.items():
            try:
                pipe.stop()
                logging.info(f"Pipeline stopped for camera {serial}")
            except Exception as e:
                logging.error(f"Failed to stop pipeline for camera {serial}: {e}")