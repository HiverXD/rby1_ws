import sys
import os
import numpy as np
from PIL import Image
from types import SimpleNamespace
import argparse
import copy

from .anygrasp_manipulation import ObjectHandler
from .utils.utils import get_3d_points, get_grasp_pose_in_world
from .utils.camera import CameraParameters

from .utils_forcali.utils import FrameCfg, auto_calibrate_frames, get_grasp_pose_in_world_calibrated
from scipy.spatial.transform import Rotation as R
FAIL_COLOR = "\033[91m"
NO_LICENSE_MSG = f"""
{FAIL_COLOR}Couldn't find the license folder in the /src directory.
Check the readme at https://github.com/graspnet/anygrasp_sdk?tab=readme-ov-file#license-registration
to get a license for yourself.

If you already have a license, group the license related .json, .lic, .public_key,
.signature files into a "license" folder and place it inside the /src directory
"""

def check_license_folder():
    license_path = os.path.join(os.path.dirname(__file__), "license")
    if (not os.path.exists(license_path)) or (len(os.listdir(license_path)) < 4):
        print(NO_LICENSE_MSG)
        sys.exit(1)


def add_predictor_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint_path",
        default="/home/usd/Research/IsaacLab_Ext_with_RBY1/ok_robot/ok_robot_manipulation/src/checkpoints/checkpoint_detection.tar",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--max_gripper_width",
        type=float,
        default=0.1,
        help="Maximum gripper width (<=0.1m)",
    )
    parser.add_argument("--gripper_height", type=float, default=0.07, help="Gripper height")
    parser.add_argument(
        "--top_down_grasp", action="store_true", help="Output top-down grasps"
    )
    parser.add_argument("--debug", action="store_true", help="Enable visualization")

    parser.add_argument("--predictor_headless", action="store_true", help="Enable predictor headless mode")
    parser.add_argument(
        "--max_depth", type=float, default=2.0, help="Maximum depth of point cloud"
    )
    parser.add_argument(
        "--min_depth", type=float, default=0, help="Minimum depth of point cloud"
    )
    parser.add_argument(
        "--sampling_rate", type=float, default=1.0, help="Sampling rate of points [<= 1]"
    )
    parser.add_argument(
        "--query", type=str, default="cube", help="Object query"
    )
    
    parser.add_argument(
        "--open_communication",
        action="store_true",
        help="Use image transferred from the robot",
    )
    return parser


class GraspPredictor:
    def __init__(self, cfgs: argparse.Namespace):

        check_license_folder()
        self.cfgs = cfgs

        self.object_handler = ObjectHandler(
            cfgs=self.cfgs,
        )
        self.frame_cfg: FrameCfg | None = None

    def predict(self, rgb_image: np.ndarray, depths: np.ndarray, head_link_pose_in_world: list | None = None, data_collect: bool = False) -> np.ndarray | None:

        # fx, fy, cx, cy, scale = 473.4348, 473.4348, 256.0, 256.0, 1.0 
        # fx, fy, cx, cy, scale = 512, 512, 256.0, 256.0, 1.0 
        fx, fy, cx, cy, scale = 455.11, 455.11, 256.0, 256.0, 1.0 
        # fx, fy, cx, cy, scale = 1024, 102, 256.0, 256.0, 1.0
        colors = np.array(rgb_image)
        
        depths = depths * scale
        cam = CameraParameters(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            head_tilt=-60,
            image=rgb_image,
            colors=colors / 255.0,
            depths=depths
        )
        self.object_handler.cam = cam
        self.object_handler.query = self.cfgs.query

        save_dir = os.path.join(self.cfgs.environment, self.cfgs.query, "anygrasp")
        os.makedirs(save_dir, exist_ok=True)
        box_filename = f"{save_dir}/object_detection.jpg"
        mask_filename = f"{save_dir}/semantic_segmentation.jpg"

        seg_mask, bbox = self.object_handler.lang_sam.detect_obj(
            cam.image,
            self.cfgs.query,
            visualize_box=True,
            visualize_mask=True,
            box_filename=box_filename,
            mask_filename=mask_filename
        )
        if bbox is None:
            print(f"not found {self.cfgs.query} in the image")
            return None
        points = get_3d_points(self.object_handler.cam)

        # data_collect=True  -> GraspGroup (Grasp 묶음)
        # data_collect=False -> Grasp (단일 Grasp)
        grasp_results = self.object_handler.pickup(points, seg_mask, bbox, False, head_link_pose_in_world, data_collect)
        
        if not grasp_results:
            return None
        if data_collect:
            grasp_candidates = grasp_results
            final_poses_to_return = []
            best_grasp_in_model_frame = []
            for grasp in grasp_candidates:
                original_pose, _, _ = get_grasp_pose_in_world(
                    grasp,
                    head_link_pose_in_world
                )
                final_poses_to_return.append(original_pose)
                best_grasp_in_model_frame.append(grasp.translation)
            return final_poses_to_return, best_grasp_in_model_frame
            
        else:
            best_grasp_in_model_frame = grasp_results
            model_transition = best_grasp_in_model_frame.translation

            original_pose, _, _ = get_grasp_pose_in_world(
                best_grasp_in_model_frame,
                head_link_pose_in_world
            )

            final_grasp_pose_world = original_pose

            # 기존과 동일하게 단일 pose와 translation을 반환합니다.
            return final_grasp_pose_world, best_grasp_in_model_frame.translation
    
    def get_object_center_in_world(self, bbox, depths, cam_params, head_link_pose_in_world):
        """탐지된 객체의 중심점의 월드 좌표를 계산합니다."""
        # 1. Bbox 중심 픽셀 좌표 계산
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
        center_pixel_x = int((bbox_x_min + bbox_x_max) / 2)
        center_pixel_y = int((bbox_y_min + bbox_y_max) / 2)

        # 2. 중심 픽셀의 깊이 값 읽기 (중심점 주변 평균을 내면 더 안정적)
        patch = depths[center_pixel_y - 2 : center_pixel_y + 3, center_pixel_x - 2 : center_pixel_x + 3]
        depth = np.mean(patch)

        if depth <= 0:
            print("[경고] 객체 중심의 유효한 깊이 값을 얻지 못했습니다.")
            return None

        # 3. 2D 픽셀 -> 3D 카메라 좌표계로 변환 (Unprojection)
        fx, fy, cx, cy = cam_params['fx'], cam_params['fy'], cam_params['cx'], cam_params['cy']
        point_in_cam_frame = np.array([
            (center_pixel_x - cx) * depth / fx,
            (center_pixel_y - cy) * depth / fy,
            depth
        ])
        
        # y축 부호가 반대인 경우 보정
        point_in_cam_frame[1] *= -1

        # 4. 3D 카메라 좌표 -> 3D 월드 좌표로 변환
        t_camera_in_world = head_link_pose_in_world[0]
        quat_camera_in_world_wxyz = head_link_pose_in_world[1]
        R_camera_in_world = R.from_quat([
            quat_camera_in_world_wxyz[1], quat_camera_in_world_wxyz[2], 
            quat_camera_in_world_wxyz[3], quat_camera_in_world_wxyz[0]
        ]).as_matrix()

        point_in_world_frame = R_camera_in_world @ point_in_cam_frame + t_camera_in_world
        
        return point_in_world_frame

