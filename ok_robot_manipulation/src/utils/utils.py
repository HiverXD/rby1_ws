import copy

import numpy as np
from PIL import Image
import open3d as o3d

from ..utils.camera import CameraParameters
from PIL import ImageDraw

from scipy.spatial.transform import Rotation as R

def quat_to_rot_matrix(q_wxyz):
    """
    (w, x, y, z) 쿼터니언을 3x3 회전 행렬로 변환합니다.
    """
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    if n == 0:
        print("Warning: Zero quaternion detected, returning identity matrix.")
        return np.identity(3)
    
    q_norm = np.array([w, x, y, z]) / np.sqrt(n)
    w, x, y, z = q_norm
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def rot_matrix_to_quat(m):
    t = np.trace(m)
    if t > 0:
        r = np.sqrt(1 + t)
        w = 0.5 * r
        x = (m[2, 1] - m[1, 2]) / (2 * r)
        y = (m[0, 2] - m[2, 0]) / (2 * r)
        z = (m[1, 0] - m[0, 1]) / (2 * r)
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            r = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            x = 0.5 * r
            w = (m[2, 1] - m[1, 2]) / (2 * r)
            y = (m[0, 1] + m[1, 0]) / (2 * r)
            z = (m[0, 2] + m[2, 0]) / (2 * r)
        elif m[1, 1] > m[2, 2]:
            r = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            y = 0.5 * r
            w = (m[0, 2] - m[2, 0]) / (2 * r)
            x = (m[0, 1] + m[1, 0]) / (2 * r)
            z = (m[1, 2] + m[2, 1]) / (2 * r)
        else:
            r = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            z = 0.5 * r
            w = (m[1, 0] - m[0, 1]) / (2 * r)
            x = (m[0, 2] + m[2, 0]) / (2 * r)
            y = (m[1, 2] + m[2, 1]) / (2 * r)
    return np.array([w, x, y, z])


def create_rotated_grasp_pose(initial_pose):
    """
    주어진 Grasp Pose를 그리퍼의 접근축(Local Z-axis) 기준으로 180도 회전시킨 
    새로운 Pose를 생성합니다.

    Args:
        initial_pose (np.array): [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] 형태의 초기 포즈.

    Returns:
        np.array: 180도 회전된 새로운 포즈.
    """
    # 1. 위치와 방향(쿼터니언) 분리
    position = initial_pose[:3]
    quat_initial_wxyz = initial_pose[3:]

    # 2. Scipy의 Rotation 객체로 변환
    # 중요: scipy는 쿼터니언을 [x, y, z, w] 순서로 다루므로 변환이 필요합니다.
    quat_initial_xyzw = np.array([
        quat_initial_wxyz[1], quat_initial_wxyz[2], quat_initial_wxyz[3], quat_initial_wxyz[0]
    ])
    R_initial = R.from_quat(quat_initial_xyzw)

    # 3. 그리퍼의 Z축 기준 180도 회전을 나타내는 Rotation 객체 생성
    R_rot_180_z = R.from_euler('z', 180, degrees=True)

    # 4. 초기 방향에 180도 회전을 적용 (쿼터니언 곱셈)
    # R_initial * R_rot_180_z는 R_initial의 로컬 좌표계에서 Z축 회전을 적용합니다.
    R_new = R_initial * R_rot_180_z

    # 5. 결과를 다시 쿼터니언 배열로 변환
    quat_new_xyzw = R_new.as_quat()
    
    # 6. 원래의 [w, x, y, z] 순서로 복원
    quat_new_wxyz = np.array([
        quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]
    ])

    # 7. 위치와 새로운 방향을 합쳐서 최종 포즈 반환
    rotated_pose = np.concatenate((position, quat_new_wxyz))
    
    return rotated_pose



def get_grasp_pose_in_world(
    model_output,
    camera_link_pose_world, # base to camera
):
    # 모델 : X-Right, Y-Down, Z-Forward
    # 카메라 : X-Right, Y-Down, Z-Forward
    # 월드 : X-Forward, Y-Left, Z-Up
    
    t_grasp_in_model = np.array(model_output.translation)
    t_grasp_in_model[1] = -t_grasp_in_model[1]
    
    R_grasp_in_model_matrix_raw = np.array(model_output.rotation_matrix)

    # model_to_rby1_gripper_matrix = np.identity(3)

    correct_ratotation_matrix = np.array([
        [ 1, 0, 0],
        [0, -1, 0],
        [ 0, 0, 1]
    ])
    
    
    model_to_rby1_gripper_matrix = np.array([
        [ 0, 0, -1],
        [ -1, 0, 0],
        [ 0, 1, 0]
    ])
    
    # match model gripper frame to rby1 gripper frame
    R_correct =  correct_ratotation_matrix @ R_grasp_in_model_matrix_raw @ correct_ratotation_matrix
    R_grasp_in_model_matrix =  R_correct @ model_to_rby1_gripper_matrix
    
    reflection_matrix = np.identity(3)

    # grasp pose in camera frame
    t_grasp_in_camera = reflection_matrix @ t_grasp_in_model 
    R_grasp_in_camera_matrix = reflection_matrix @ R_grasp_in_model_matrix


    t_camera_in_world = np.array(camera_link_pose_world[0])
    quat_camera_in_world_wxyz = np.array(camera_link_pose_world[1])
    R_camera_in_world_matrix = quat_to_rot_matrix(quat_camera_in_world_wxyz)

    # grasp pose in world
    R_final_in_world_matrix = R_camera_in_world_matrix @ R_grasp_in_camera_matrix
    t_offset_in_world = R_camera_in_world_matrix @ t_grasp_in_camera

    t_final_in_world = t_camera_in_world + t_offset_in_world
    final_quat_wxyz = rot_matrix_to_quat(R_final_in_world_matrix)

    # print(f"t_final_in_world: {t_final_in_world}")
    # rby1 gripper frame 기준 offset
    # tcp_offset_in_gripper_frame1 = np.array([0, 0.015, 0])
    # tcp_offset_in_world_frame1 = R_final_in_world_matrix @ tcp_offset_in_gripper_frame1
    tcp_offset_in_gripper_frame2 = np.array([0, 0.0, 0.0])
    tcp_offset_in_world_frame2 = R_final_in_world_matrix @ tcp_offset_in_gripper_frame2
    
    t_final_for_robot_origin = t_final_in_world  + tcp_offset_in_world_frame2 # + tcp_offset_in_world_frame1
    original_pose = np.concatenate((t_final_for_robot_origin, final_quat_wxyz))
    rotated_pose = create_rotated_grasp_pose(original_pose)

    gripper_y_axis_in_world = R_final_in_world_matrix[:, 1] 
    
    # 그리퍼의 Z축의 월드 Z 성분 값 (월드 Z축은 'Up' 방향)
    # 이 값이 양수(+)이면 손등이 위를, 음수(-)이면 손등이 아래를(뒤집힘) 향합니다.
    z_component_of_gripper_y = gripper_y_axis_in_world[2]
    
    # # 만약 그리퍼가 뒤집혀 있다면(손등이 아래를 향한다면), 180도 회전된 자세를 선택
    # if z_component_of_gripper_y < 0:
    #     print("[INFO] Gripper orientation is upside down. Choosing the 180-degree rotated pose.")
    #     return rotated_pose
    # else:
    #     print("[INFO] Gripper orientation is upright. Choosing the original pose.")
    #     return original_pose
    return original_pose, rotated_pose, R_final_in_world_matrix
    
    
def sample_points(points, sampling_rate=1):
    N = len(points)
    num_samples = int(N*sampling_rate)
    indices = np.random.choice(N, num_samples, replace=False)
    sampled_points = points[indices]
    return sampled_points, indices

def get_3d_points(cam: CameraParameters):

    xmap, ymap = np.arange(cam.depths.shape[1]), np.arange(cam.depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = cam.depths
    points_x = (xmap - cam.cx) / cam.fx * points_z
    points_y = (ymap - cam.cy) / cam.fy * points_z

    points = np.stack((points_x, points_y, points_z), axis=2)
    return points

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def draw_seg_mask(image, seg_mask, save_file=None):
    alpha = np.where(seg_mask > 0, 128, 0).astype(np.uint8)

    image_pil = copy.deepcopy(image)
    alpha_pil = Image.fromarray(alpha)
    image_pil.putalpha(alpha_pil)

    if save_file is not None:
        image_pil.save(save_file)
        print(f"Saved Segementation Mask at {save_file}")

def draw_rectangle(image, bbox, width=5):
    img_drw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    width_increase = 5
    for _ in range(width_increase):
        img_drw.rectangle([(x1, y1), (x2, y2)], outline="green")

        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
    
    return img_drw

def color_grippers(grippers, max_score, min_score):
    """
        grippers    : list of grippers of form graspnetAPI grasps
        max_score   : max score of grippers
        min_score   : min score of grippers

        For debugging purpose - color the grippers according to score
    """

    for idx, gripper in enumerate(grippers):
        g = grippers[idx]
        if max_score != min_score:
            color_val = (g.score - min_score)/(max_score - min_score)
        else:
            color_val = 1
        color = [color_val, 0, 0]
        print(g.score, color)
        gripper.paint_uniform_color(color)

    return grippers

def visualize_cloud_geometries(cloud, geometries, translation = None, rotation = None, visualize = True, save_file = None):
    """
        cloud       : Point cloud of points
        grippers    : list of grippers of form graspnetAPI grasps
        visualise   : To show windows
        save_file   : Visualisation file name
    """

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    if translation is not None:
        coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        translation[2] = -translation[2]
        coordinate_frame1.translate(translation)
        coordinate_frame1.rotate(rotation)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=visualize)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.add_geometry(cloud)
    if translation is not None:
        visualizer.add_geometry(coordinate_frame1)
    visualizer.poll_events()
    visualizer.update_renderer()

    if save_file is not None:
        ## Controlling the zoom
        view_control = visualizer.get_view_control()
        zoom_scale_factor = 1.4  
        view_control.scale(zoom_scale_factor)

        visualizer.capture_screen_image(save_file, do_render = True)
        print(f"Saved screen shot visualization at {save_file}")

    if visualize:
        visualizer.add_geometry(coordinate_frame)
        visualizer.run()
    else:
        visualizer.destroy_window()    
# import open3d as o3d
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import io
# from PIL import Image # Pillow는 이미 사용 중이실 겁니다.
# def create_colorbar_image(cmap_name, num_steps, size=(60, 250)):
#     """Matplotlib을 사용해 컬러바 이미지를 생성하고 Pillow 이미지 객체로 반환합니다."""
    
#     # Matplotlib 백엔드를 AGG로 설정하여 화면에 표시하지 않고 작업
#     matplotlib.use('Agg')
    
#     fig = plt.figure(figsize=(size[0]/100.0, size[1]/100.0), dpi=100)
#     ax = fig.add_axes([0.15, 0.05, 0.2, 0.9]) # [left, bottom, width, height]

#     cmap = plt.get_cmap(cmap_name)
#     # 0 (최고 순위) 부터 num_steps-1 (최저 순위) 까지의 범위를 정규화
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=num_steps - 1)
    
#     # 컬러바 생성
#     cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
#     cb.set_label('Grasp Rank (Higher is better)')
    
#     # 이미지를 메모리 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', transparent=True)
#     buf.seek(0)
    
#     # Matplotlib figure를 닫아 리소스 정리
#     plt.close(fig)
    
#     # 버퍼에서 Pillow 이미지로 변환하여 반환
#     return Image.open(buf)
# def visualize_cloud_geometries(cloud, geometries, translation = None, rotation = None, visualize = True, save_file = None, cmap_name='jet_r'):
#     """
#         cloud       : Point cloud of points
#         grippers    : list of grippers of form graspnetAPI grasps
#         visualise   : To show windows
#         save_file   : Visualisation file name
#     """

#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
#     if translation is not None:
#         coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
#         translation[2] = -translation[2]
#         coordinate_frame1.translate(translation)
#         coordinate_frame1.rotate(rotation)

#     visualizer = o3d.visualization.Visualizer()
#     visualizer.create_window(visible=visualize)
#     for geometry in geometries:
#         visualizer.add_geometry(geometry)
#     visualizer.add_geometry(cloud)
#     if translation is not None:
#         visualizer.add_geometry(coordinate_frame1)
#     visualizer.poll_events()
#     visualizer.update_renderer()

#     if save_file is not None:
#         # 1. Open3D 뷰를 Pillow 이미지로 캡처
#         image_buffer = visualizer.capture_screen_float_buffer(do_render=True)
#         image_np = (np.asarray(image_buffer) * 255).astype(np.uint8)
#         main_image = Image.fromarray(image_np)

#         # 2. 컬러바 범례 이미지 생성
#         num_grasps = len(geometries)
#         if num_grasps > 0:
#             colorbar_image = create_colorbar_image(cmap_name, num_grasps, size=(80, main_image.height // 2))

#             # 3. 메인 이미지에 컬러바 이미지 합성 (오른쪽 상단에 배치)
#             margin = 10
#             position = (main_image.width - colorbar_image.width - margin, margin)
            
#             # 투명 배경을 위해 마스크 사용
#             main_image.paste(colorbar_image, position, colorbar_image)

#         # 4. 최종 합성된 이미지 저장
#         main_image.save(save_file)
#         print(f"Saved screen shot with colorbar at {save_file}")
#     if visualize:
#         visualizer.add_geometry(coordinate_frame)
#         visualizer.run()
#     else:
#         visualizer.destroy_window()