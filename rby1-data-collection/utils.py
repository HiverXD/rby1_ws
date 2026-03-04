import numpy as np
import os
import yaml
import logging
import rby1_sdk as rby
from typing import Union
import cv2
import h5py

try:
    from sensor_msgs.msg import Image
except ImportError:
    Image = None  # ROS not available; rosimg_to_numpy() will not be usable

def rosimg_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image → np.ndarray (HxWxC, uint8)."""
    dtype = np.uint8  # adjust if msg.encoding differs
    img = np.frombuffer(msg.data, dtype=dtype)
    img = img.reshape((msg.height, msg.width, -1))  # works for RGB/BGR/mono # (H, W, C)
    return img


def get_next_h5_path(base_dir="/home/nvidia/rby1_ws/rby1-data-collection/Demo"):
# def get_next_h5_path(base_dir="/media/nvidia/T7/Demo"):
    os.makedirs(base_dir, exist_ok=True)

    # 기존 episode 폴더 번호 중 최댓값 + 1을 다음 인덱스로 사용.
    # len() 대신 max()를 사용하는 이유:
    #   Discard 등으로 중간 번호가 삭제될 경우 len() == 기존 번호와 충돌하여 덮어씌우는 버그 방지.
    existing_indices = []
    for d in os.listdir(base_dir):
        if d.startswith("episode_") and os.path.isdir(os.path.join(base_dir, d)):
            suffix = d[len("episode_"):]
            if suffix.isdigit():
                existing_indices.append(int(suffix))

    next_index = (max(existing_indices) + 1) if existing_indices else 0

    # Create episode folder
    episode_dir = os.path.join(base_dir, f"episode_{next_index}")
    os.makedirs(episode_dir, exist_ok=True)

    # Create path for h5 file inside episode folder
    return os.path.join(episode_dir, f"demo_{next_index}.h5")


def elbows_bending_check(robot: rby.Robot_A) -> bool:
    """
    Checks if elbow angles exceed a threshold to determine if a movement should be skipped.

    This function encapsulates the entire logic:
    1. Reads the current joint angles from the robot state.
    2. Gets robot model details (DOFs).
    3. Reads the angle threshold from 'config.yaml'.
    4. Compares the current elbow angles against the threshold.

    Args:
        robot: The robot object from which to get the state.

    Returns:
        True if the movement should be skipped (angle exceeded), False otherwise.
    """
    if robot is None:
        return False
        
    robot_state = robot.get_state()
    if robot_state is None:
        return False

    joint_angles = robot_state.position
    if joint_angles is None or joint_angles.size == 0:
        # If joint data is unavailable, assume no skip for now.
        # Depending on safety requirements, this might need to raise an error or return True to force a skip.
        return False

    try:
        # Get DOFs from robot model
        model = robot.model()
        head_dof = len(model.head_idx)
        torso_dof = len(model.torso_idx)
        right_arm_dof = len(model.right_arm_idx)

        # Extract elbow angles using hardcoded offsets.
        # This assumes a fixed joint order. A more robust solution would dynamically
        # look up joint indices by name if the SDK provides such functionality.
        right_elbow_angle = joint_angles[head_dof + torso_dof + 3]
        left_elbow_angle = joint_angles[head_dof + torso_dof + right_arm_dof + 3]

        # Load threshold from YAML
        try:
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
            with open(_cfg_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            elbow_threshold_deg = config.get('elbow_angle_threshold_deg', 90)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load or parse config.yaml: {e}. Using default threshold 90 deg.")
            elbow_threshold_deg = 90

        elbow_threshold_rad = np.deg2rad(elbow_threshold_deg)

        # Perform the check
        if abs(right_elbow_angle) > elbow_threshold_rad or abs(left_elbow_angle) > elbow_threshold_rad:
            return True

    except IndexError:
        logging.warning("Joint angle index out of bounds when checking elbow angles. Check joint_angles size and DOF values.")
    except Exception as e:
        logging.warning(f"An unexpected error occurred in check_elbows_for_skip: {e}")

    return False

def save_video(h5_path, output_filename="robot_video.avi", camera_name='head', fps=30):
    if not os.path.exists(h5_path):
        logging.warning(f"❌ 파일을 찾을 수 없습니다.")
        return

    rgb_key = f'{camera_name}_rgb'
    depth_key = f'{camera_name}_depth'

    # H5 파일이 있는 '폴더 경로' 추출
    # 예: /home/user/Task/episode1/data.h5 -> /home/user/Task/episode1
    dir_path = os.path.dirname(os.path.abspath(h5_path))
    
    # 저장할 전체 경로 생성
    # 예: /home/user/Task/episode1 + video.avi
    save_path = os.path.join(dir_path, output_filename)

    with h5py.File(h5_path, 'r') as f:
        if rgb_key not in f:
            logging.warning("❌ 데이터가 없습니다.")
            return

        # 1. 데이터 로드 (전체 프레임)
        rgb_data = f[f'{rgb_key}/image'][:]
        has_depth = (depth_key in f)
        if has_depth:
            depth_data = f[f'{depth_key}/image'][:]
        
        num_frames, height, width, _ = rgb_data.shape
        
        # 2. 화면 크기 설정 (Depth 있으면 가로 2배)
        output_width = width * 2 if has_depth else width
        
        # 3. 비디오 작성자 설정 (MJPG 코덱 사용 -> 안전함)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, height))

        # 4. 프레임 쓰기
        for i in range(num_frames):
            img = rgb_data[i]
            # RGB -> BGR 변환 (OpenCV 저장용)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if has_depth:
                d_img = depth_data[i]
                # Depth 컬러 입히기
                d_norm = cv2.normalize(d_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                
                # 합치기
                combined = np.hstack((img_bgr, d_color))
                out.write(combined)
            else:
                out.write(img_bgr)
        
        out.release()

    logging.info(f"✅ 저장 완료! 파일 위치: {save_path}")