#!/usr/bin/env python3
"""
H5 형식의 데이터를 LeRobot 형식으로 변환하는 스크립트

사용법:
    python convert_to_lerobot.py [--h5_path <h5_file_path>] [--output_dir <output_dir>] [--repo_id <repo_id>] [--fps <fps>] [--task <task>]
    
예시:
    # 기본값 사용 (repo_id는 자동으로 "rby1_teleop_demo" 사용)
    python convert_to_lerobot.py --h5_path /media/nvidia/T7/Demo/demo_104.h5
    
    # repo_id 명시적으로 지정
    python convert_to_lerobot.py --h5_path /media/nvidia/T7/Demo/demo_104.h5 --repo_id my_custom_dataset
    
    # task 지정
    python convert_to_lerobot.py --h5_path /media/nvidia/T7/Demo/demo_104.h5 --task "pick_and_place"
"""

import argparse
import h5py
import numpy as np
import torch
import logging
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_nearest_index(times: np.ndarray, target_time: float) -> int:
    """
    타겟 타임스탬프에 가장 가까운 인덱스를 찾습니다.
    
    동기화가 필요한 이유:
    - H5 파일에서 samples 그룹(로봇 관절 데이터)과 head_rgb 그룹(이미지)은
      서로 다른 타임스탬프를 가질 수 있습니다.
    - 예: 로봇 데이터는 30fps, 카메라는 10fps로 기록될 수 있습니다.
    - 각 샘플에 대해 가장 가까운 시간의 이미지를 찾아 매칭해야 합니다.
    """
    return np.argmin(np.abs(times - target_time))


def convert_h5_to_lerobot(
    h5_path: str,
    output_dir: str,
    repo_id: Optional[str] = None,
    fps: int = 10,
    robot_type: str = "rby1",
    task: str = ""
):
    """
    H5 파일을 LeRobot 형식으로 변환합니다.
    
    Args:
        h5_path: 입력 H5 파일 경로
        output_dir: 출력 디렉터리 경로
        repo_id: LeRobot 저장소 ID (None이면 기본값 "rby1_teleop_demo" 사용)
        fps: 프레임 레이트
        robot_type: 로봇 타입
        task: task feature 값 (기본값: 빈 문자열)
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 파일을 찾을 수 없습니다: {h5_path}")
    
    # repo_id가 지정되지 않으면 기본값 사용
    # repo_id는 LeRobot에서 데이터셋을 구분하는 식별자입니다.
    # 하나의 output_dir 아래에 여러 데이터셋을 저장할 수 있도록 합니다.
    # 예: output_dir/repo_id/episodes/... 형태로 저장됩니다.
    if repo_id is None:
        repo_id = "rby1_teleop_demo"
        logging.info(f"repo_id가 지정되지 않아 기본값 사용: {repo_id}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LeRobot은 root/repo_id 구조를 사용합니다
    # lerobot_handler.py와 동일하게 root_dir/repo_id를 root로 사용
    root_path = output_dir / repo_id
    
    logging.info(f"H5 파일 읽기: {h5_path}")
    logging.info(f"출력 디렉터리: {output_dir}")
    logging.info(f"LeRobot repo_id: {repo_id}")
    logging.info(f"LeRobot root 경로: {root_path}")
    
    with h5py.File(h5_path, "r") as f:
        # samples 그룹 읽기
        samples_grp = f.get("samples")
        if samples_grp is None:
            raise ValueError("H5 파일에 'samples' 그룹이 없습니다.")
        
        sample_times = samples_grp["time"][()] if "time" in samples_grp else None
        robot_positions = samples_grp["robot_position"][()] if "robot_position" in samples_grp else None
        robot_target_joints = samples_grp["robot_target_joints"][()] if "robot_target_joints" in samples_grp else None
        gripper_states = samples_grp["gripper_state"][()] if "gripper_state" in samples_grp else None
        
        if robot_positions is None:
            raise ValueError("H5 파일에 'robot_position' 데이터가 없습니다.")
        
        num_samples = len(robot_positions)
        logging.info(f"샘플 수: {num_samples}")
        
        # head_rgb 그룹 읽기
        # 동기화가 필요한 이유: samples 그룹의 데이터와 이미지가 서로 다른 타임스탬프를 가질 수 있음
        head_rgb_grp = f.get("head_rgb")
        rgb_times = None
        rgb_images = None
        if head_rgb_grp is not None:
            rgb_times = head_rgb_grp["time"][()] if "time" in head_rgb_grp else None
            rgb_images = head_rgb_grp["image"][()] if "image" in head_rgb_grp else None
            if rgb_images is not None:
                logging.info(f"RGB 이미지 수: {len(rgb_images)}")
                if sample_times is not None and rgb_times is not None:
                    logging.info(f"샘플 타임스탬프 범위: {sample_times[0]:.3f} ~ {sample_times[-1]:.3f}")
                    logging.info(f"RGB 타임스탬프 범위: {rgb_times[0]:.3f} ~ {rgb_times[-1]:.3f}")
                    logging.info("타임스탬프 기반 동기화를 사용합니다.")
        
        # 첫 번째 샘플로 데이터셋 초기화
        first_sample = {
            "robot_position": robot_positions[0],
            "robot_target_joints": robot_target_joints[0] if robot_target_joints is not None else robot_positions[0],
            "gripper_state": gripper_states[0] if gripper_states is not None else np.zeros(2),
            "head_rgb": rgb_images[0] if rgb_images is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            "task": task,
        }
        
        # LeRobot 데이터셋 생성
        rp_dim = len(first_sample["robot_position"])
        gs_dim = len(first_sample["gripper_state"])
        
        features = {
            "observation.state": {"dtype": "float32", "shape": (rp_dim,)},
            "action": {"dtype": "float32", "shape": (rp_dim,)},
            "observation.gripper_state": {"dtype": "float32", "shape": (gs_dim,)},
            "observation.images.head_rgb": {"dtype": "video", "shape": (480, 640, 3)},
        }
        
        logging.info("LeRobot 데이터셋 생성 중...")
        # 기존 데이터셋이 있으면 삭제 (새로 생성하기 위해)
        if root_path.exists():
            logging.warning(f"기존 데이터셋이 존재합니다: {root_path}. 삭제하고 새로 생성합니다.")
            shutil.rmtree(root_path)
        
        # 새 데이터셋 생성
        # lerobot_handler.py와 동일하게 root_dir/repo_id를 root로 사용
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root_path,
            features=features,
            fps=fps,
            robot_type=robot_type,
        )
        
        # 데이터 변환 및 추가
        logging.info("데이터 변환 및 저장 중...")
        for i in range(num_samples):
            if i % 10 == 0:
                logging.info(f"진행 중: {i}/{num_samples} ({100*i/num_samples:.1f}%)")
            
            # 기본 데이터
            frame = {
                "observation.state": torch.from_numpy(robot_positions[i]).float(),
                "action": torch.from_numpy(robot_target_joints[i] if robot_target_joints is not None else robot_positions[i]).float(),
                "observation.gripper_state": torch.from_numpy(gripper_states[i] if gripper_states is not None else np.zeros(gs_dim)).float(),
                "task": task,  # task feature (인자로 받은 값 사용)
            }
            
            # RGB 이미지 동기화
            # 동기화 이유: samples 그룹의 데이터와 이미지가 서로 다른 타임스탬프를 가질 수 있음
            # 예: 로봇 데이터는 30fps, 카메라는 10fps로 기록될 수 있음
            # 각 샘플에 대해 가장 가까운 시간의 이미지를 찾아 매칭
            if rgb_images is not None and rgb_times is not None and sample_times is not None:
                # 타임스탬프 기반 동기화: 가장 가까운 시간의 이미지 찾기
                rgb_idx = find_nearest_index(rgb_times, sample_times[i])
                frame["observation.images.head_rgb"] = torch.from_numpy(rgb_images[rgb_idx])
            elif rgb_images is not None:
                # 타임스탬프가 없으면 인덱스로 매칭 (길이가 같다고 가정)
                if i < len(rgb_images):
                    frame["observation.images.head_rgb"] = torch.from_numpy(rgb_images[i])
                else:
                    frame["observation.images.head_rgb"] = torch.from_numpy(rgb_images[-1])
            else:
                # RGB 이미지가 없으면 빈 이미지
                frame["observation.images.head_rgb"] = torch.zeros((480, 640, 3), dtype=torch.uint8)
            
            dataset.add_frame(frame)
        
        # 에피소드 저장
        logging.info("에피소드 저장 중...")
        dataset.save_episode()
        logging.info(f"변환 완료! 저장 위치: {root_path}")


def main():
    parser = argparse.ArgumentParser(
        description="H5 형식의 데이터를 LeRobot 형식으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/media/nvidia/T7/Demo/demo_104.h5",
        help="입력 H5 파일 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/nvidia/T7/Demo/LeRobotData",
        help="출력 디렉터리 경로 (기본값: /media/nvidia/T7/Demo/LeRobotData)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="rby1_teleop_demo",
        help="LeRobot 저장소 ID (지정하지 않으면 기본값 'rby1_teleop_demo' 사용)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="프레임 레이트 (기본값: 10)"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="rby1",
        help="로봇 타입 (기본값: rby1)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="task feature 값 (기본값: 빈 문자열)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_h5_to_lerobot(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type=args.robot_type,
            task=args.task
        )
    except Exception as e:
        logging.error(f"변환 중 오류 발생: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

