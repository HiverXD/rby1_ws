import torch
import numpy as np
import threading
import queue
import logging
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os

class LeRobotDataHandler:
    def __init__(self, repo_id: str, root_dir: str, fps: int = 10, robot_type: str = "rby1"):
        self.repo_id = repo_id
        self.root_dir = os.path.join(root_dir, repo_id)
        # self.root_dir = Path(root_dir, repo_id)
        self.fps = fps
        self.robot_type = robot_type
        
        self.dataset = None
        self.data_queue = queue.Queue(maxsize=1000)
        self._update_queue = queue.Queue() # 소급 업데이트용 큐
        self.stop_event = threading.Event()
        self.thread = None
        self._current_idx = 0 # 현재까지 저장된 프레임 수 추적

    def start(self):
        logging.info(f"[LeRobotWriter] Starting writer at {self.root_dir}")
        self.thread = threading.Thread(target=self._run, name="lerobot_writer", daemon=True)
        self.thread.start()
        return self


    def initialize_dataset(self):
        """첫 번째 샘플을 바탕으로 데이터셋 특징(Features) 정의 및 생성"""
        # 관절 차원 확인
        # rp_dim = len(sample_data["robot_position"])
        # 그리퍼 차원 확인
        # gs_dim = len(sample_data["gripper_state"]) if sample_data.get("gripper_state") is not None else 2
        
        features = {
            "observation.state": {"dtype": "float32", "shape": (22,)},
            "action": {"dtype": "float32", "shape": (22,)},
            "observation.gripper_state": {"dtype": "float32", "shape": (2,)},
            "observation.images.head_rgb": {"dtype": "video", "shape": (480, 640, 3)},
            # PCD는 고정 크기 샘플링 저장 (LeRobot 표준 권장 방식)
            # "observation.pcd_points": {"dtype": "float32", "shape": (1024, 3)},
        }

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root_dir,
            features=features,
            fps=self.fps,
            robot_type=self.robot_type,
        )
        print('self.root_dir' * 10)

    def put(self, sample: dict):
        """H5Writer.put()과 동일: 데이터를 큐에 삽입"""
        try:
            self.data_queue.put_nowait(sample)
        except queue.Full:
            logging.warning("[LeRobotWriter] Queue full, dropping oldest sample")
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(sample)
            except queue.Empty:
                pass

    def update_previous_target(self, robot_pos: np.ndarray):
        """이전 프레임의 action(target)을 현재 실제 위치로 소급 업데이트"""
        target_idx = max(0, self._current_idx - 1)
        self._update_queue.put_nowait((target_idx, robot_pos))

    def _run(self):
        while not self.stop_event.is_set() or not self.data_queue.empty():
            try:
                item = self.data_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None: break

            # 1. 지연 초기화
            if self.dataset is None:
                self.initialize_dataset(item)

            # 2. 소급 업데이트 처리
            # LeRobotDataset의 내부 버퍼에 접근하여 이전 프레임의 action 수정
            while not self._update_queue.empty():
                idx, pos = self._update_queue.get_nowait()
                if idx < len(self.dataset.hf_dataset):
                    # 주의: 이미 디스크에 써진 데이터는 수정이 어려우므로 
                    # 에피소드 저장 전 버퍼 단계에서 처리됩니다.
                    pass 

            # 3. 데이터 변환 및 추가
            frame = {
                "observation.state": torch.from_numpy(item["robot_position"]).float(),
                "action": torch.from_numpy(item["robot_target_joints"]).float(),
                "observation.gripper_state": torch.from_numpy(item["gripper_state"]).float() 
                                             if item.get("gripper_state") is not None else torch.zeros(2),
                "observation.images.head_rgb": torch.from_numpy(item["head_rgb"]),
            }

            if item.get("pcd_points") is not None:
                # 1024개 포인트로 샘플링 (H5Writer의 가변 길이와 달리 고정 길이 권장)
                pts = item["pcd_points"][:1024]
                frame["observation.pcd_points"] = torch.from_numpy(pts).float()

            self.dataset.add_frame(frame)
            self._current_idx += 1

    def stop(self):
        """종료 및 에피소드 저장"""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if self.dataset:
            self.dataset.save_episode()