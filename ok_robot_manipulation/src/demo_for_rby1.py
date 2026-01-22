import argparse
import sys
import os

from .anygrasp_manipulation import ObjectHandler


import numpy as np
from PIL import Image


from utils.utils import get_3d_points, get_grasp_pose_in_world
from utils.camera import CameraParameters

from scipy.spatial.transform import Rotation as R

FAIL_COLOR = "\033[91m"
NO_LICENSE_MSG = f"""
{FAIL_COLOR}Couldn't find the license folder in the /src directory. 
Check the readme at https://github.com/graspnet/anygrasp_sdk?tab=readme-ov-file#license-registration 
to get a license for yourself.

If you already have a license, group the license related .json, .lic, .public_key, 
.signature files into a "license" folder and place it inside the /src directory
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    default="./checkpoints/checkpoint_detection.tar",
    help="Model checkpoint path",
)
parser.add_argument(
    "--max_gripper_width",
    type=float,
    default=0.1,
    help="Maximum gripper width (<=0.1m)",
)
parser.add_argument("--gripper_height", type=float, default=0.07, help="Gripper height")
parser.add_argument("--port", type=int, default=5556, help="port")
parser.add_argument(
    "--top_down_grasp", action="store_true", help="Output top-down grasps"
)
parser.add_argument("--debug", action="store_true", help="Enable visualization")
parser.add_argument("--headless", action="store_true", help="Enable headless mode")
parser.add_argument(
    "--open_communication",
    action="store_true",
    help="Use image transferred from the robot",
)
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
    "--query", type=str, default="black cup", help="Object query"
)

parser.add_argument("--environment", default="./example_data_rby1", help="Environment name")
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))


def check_license_folder():
    license_path = "./license"
    if (not os.path.exists(license_path)) or (len(os.listdir(license_path)) < 4):
        print(NO_LICENSE_MSG)
        sys.exit(1)


def demo(cfgs):
    # Checking the proper license folder placement.
    check_license_folder()

    object_handler = ObjectHandler(cfgs)
    
    save_dir = os.path.join(cfgs.environment, cfgs.query, "anygrasp")
    os.makedirs(save_dir, exist_ok=True)
    print(f"결과물을 '{save_dir}' 경로에 저장합니다.")

    data_dir = cfgs.environment
    try:
        color_image = Image.open(os.path.join(data_dir, "rgb.png")).convert('RGB')
        colors = np.array(color_image)
        depths = np.load(os.path.join(data_dir, "head_camera_depth.npy")) 
        # depths = np.array(Image.open(os.path.join(data_dir, "depth.png")))
        # intrinsics = np.loadtxt(os.path.join(data_dir, "intrinsics.txt"))
        # fx, fy, cx, cy, scale = intrinsics
        
        #focal_x = height * focal_length / vert_aperture
        #focal_y = width * focal_length / horiz_aperture
        #center_x = height * 0.5
        #center_y = width * 0.5




        # ROS convention (x, y, z, w)에 맞춘 쿼터니언
        quat_xyzw = [0.664463, -0.664463, 0.2418448, -0.2418448]

        r = R.from_quat(quat_xyzw)
        euler_angles = r.as_euler('xyz', degrees=True)

        print(f"Roll: {euler_angles[0]:.2f}, Pitch: {euler_angles[1]:.2f}, Yaw: {euler_angles[2]:.2f}")
        # Roll: 179.99, Pitch: -90.00, Yaw: -89.99



        print(f"depths: {depths}")
        fx, fy, cx, cy, scale = 473, 473, 256.0, 256.0, 1.0
        # fx, fy, cx, cy, scale = 306.0, 306.0, 118.0, 211.0, 1.0
        depths = depths * scale

    except FileNotFoundError:
        print(f"오류: {data_dir} 폴더에 rgb.png, depth.png, intrinsics.txt 파일이 모두 있는지 확인해주세요.")
        sys.exit(1)

    head_tilt = -45
    
        
    cam = CameraParameters(fx, fy, cx, cy, head_tilt, color_image, colors / 255.0, depths)
    object_handler.cam = cam
    object_handler.query = cfgs.query
    print(f"'{cfgs.query}' 객체를 탐지합니다...")

    box_filename = f"{save_dir}/object_detection.jpg"
    mask_filename = f"{save_dir}/semantic_segmentation.jpg"

    seg_mask, bbox = object_handler.lang_sam.detect_obj(
        cam.image,
        cfgs.query,
        visualize_box=True,
        visualize_mask=True,
        box_filename=box_filename,
        mask_filename=mask_filename
    )

    if bbox is None:
        print(f"오류: 이미지에서 '{cfgs.query}' 객체를 찾지 못했습니다.")
        sys.exit(1)
        
    print(f"객체를 탐지했습니다. BBox: {bbox}")
    print("탐지 결과(output_bbox.jpg, output_mask.jpg)가 저장되었습니다.")

    # 5. 3D 포인트 클라우드 생성
    points = get_3d_points(object_handler.cam)

    # 6. Grasp Pose 예측
    # ObjectHandler의 pickup 메소드를 호출합니다. 이 메소드는 내부적으로 AnyGrasp 모델을 사용합니다.
    # 성공 시 True를 반환하지만, 여기서는 예측된 grasp 자체를 봐야 합니다.
    # 소켓 통신을 제거했으므로, pickup 함수를 수정하여 pose를 직접 반환하게 하거나,
    # 디버깅 출력을 통해 확인해야 합니다.
    # 제공된 코드에서는 pickup 함수가 최종 pose를 소켓으로 보내므로, 여기서는 성공 여부만 확인합니다.
    # (결과를 보려면 anygrasp_manipulation.py의 pickup 함수 마지막 부분을 수정해야 합니다.)
    
    print("\nGrasp Pose를 예측합니다...")
    # pickup 함수를 호출하여 grasp을 예측합니다.
    # 마지막 인자 crop_flag는 False로 둡니다.
    best_grasp_in_cam = object_handler.pickup(points, seg_mask, bbox, False)

    if best_grasp_in_cam:
        print("\n✅ 카메라 좌표계 기준 Grasp Pose 예측에 성공했습니다!")
        
        # --- 좌표계 변환 시작 ---
        
        # ❗ 중요: 시뮬레이션 환경에서 'link_head_2'의 월드 좌표계 기준 Pose를 가져와야 합니다.
        # 이 값은 시뮬레이션 루프에서 동적으로 얻어와야 합니다. 여기서는 예시값을 사용합니다.
        # 예시: head_link가 월드 원점에서 x축으로 0.5m 이동하고 z축으로 1.0m 위에 있으며, 회전은 없다고 가정
        head_link_pose_in_world = [
            np.array([0.5, 0.0, 1.0]),          # position (x,y,z)
            np.array([1.0, 0.0, 0.0, 0.0])       # orientation (qw, qx, qy, qz) - 회전 없음
        ]

        # CameraCfg에 정의된 카메라 오프셋 정보
        # pos=(0.12, 0.0, 0.1)
        # rot=(-0.2418448, 0.664463, -0.664463, 0.2418448), convention="ros" -> (qx, qy, qz, qw)
        camera_offset_from_head = [
            np.array([0.12, 0.0, 0.1]),                                      # position (x,y,z)
            np.array([-0.2418448, 0.664463, -0.664463, 0.2418448])           # orientation (qx, qy, qz, qw)
        ]

        # 변환 함수 호출
        final_grasp_pose = get_grasp_pose_in_world(best_grasp_in_cam, head_link_pose_in_world, camera_offset_from_head)

        print("\n🌍 월드 좌표계 기준 최종 Grasp Pose:")
        print(f"  - Position (x, y, z): {final_grasp_pose[0]:.4f}, {final_grasp_pose[1]:.4f}, {final_grasp_pose[2]:.4f}")
        print(f"  - Quaternion (qw, qx, qy, qz): {final_grasp_pose[3]:.4f}, {final_grasp_pose[4]:.4f}, {final_grasp_pose[5]:.4f}, {final_grasp_pose[6]:.4f}")

        # 파일에 저장 (x,y,z,qw,qx,qy,qz 순서)
        pose_filename = os.path.join(save_dir, "grasp_pose_world.txt")
        np.savetxt(pose_filename, final_grasp_pose, fmt='%.8f', header="x y z qw qx qy qz")
        print(f"\n결과를 '{pose_filename}' 파일에 저장했습니다.")

    else:
        print("\n❌ Grasp Pose를 찾지 못했습니다. 다른 객체나 파라미터를 시도해보세요.")
if __name__ == "__main__":
    cfgs = parser.parse_args()
    demo(cfgs)
