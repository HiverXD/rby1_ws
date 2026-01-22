# ========= PATCH 1: Frame auto-calibration utilities (WXYZ action) =========
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import numpy as np

@dataclass
class FrameCfg:
    name: str
    P: np.ndarray                  # 모델(M) -> 로봇카메라(C) 3x3
    use_link_to_optical: bool      # camera_link -> camera_optical 적용 여부

def candidate_Ps():
    # 하드웨어가 성공적으로 쓴 치환 포함 (point = [-y,-x,z])
    return [
        ("I",      np.eye(3)),
        ("HW",     np.array([[0,-1,0],[-1,0,0],[0,0,1]], dtype=float)),
        ("XYswap", np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=float)),
        ("XYneg",  np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=float)),
    ]

def _score_pose(t_w: np.ndarray, R_w_tcp: R, object_pos_w: np.ndarray) -> float:
    # 거리(우선) + TCP 접근축 정합(보조). URDF 기준 접근축 = TCP -Z
    dist = np.linalg.norm(t_w - object_pos_w)
    approach_w = R_w_tcp.apply([0,0,-1])   # TCP -Z
    to_obj = object_pos_w - t_w
    n = np.linalg.norm(to_obj) + 1e-9
    cosang = np.dot(approach_w/np.linalg.norm(approach_w), to_obj/n)
    return dist + 0.05*(1 - cosang)

def _link_to_optical_rot():
    # ROS 일반 컨벤션: camera_link -> camera_optical
    return R.from_euler('z', 90, degrees=True) * R.from_euler('x', -90, degrees=True)

def try_calib_combo(model_output, cam_link_pose_w, object_pos_w, use_link_to_opt):
    t_m = np.asarray(model_output.translation)
    R_m = R.from_matrix(np.asarray(model_output.rotation_matrix))
    cam_pos_w = np.asarray(cam_link_pose_w[0])
    qw, qx, qy, qz = cam_link_pose_w[1]  # wxyz
    R_w_link = R.from_quat([qx, qy, qz, qw])  # XYZW

    R_w_cam = R_w_link * (_link_to_optical_rot() if use_link_to_opt else R.identity())

    best = None
    for name, P in candidate_Ps():
        P_R = R.from_matrix(P)
        # 모델(M) -> 카메라(C)
        t_c = P.dot(t_m)
        R_c_grasp = P_R * R_m
        # 카메라(C) -> 월드(W)
        t_w = cam_pos_w + R_w_cam.apply(t_c)
        R_w_grasp = R_w_cam * R_c_grasp
        # 모델 그립 프레임 -> TCP 정렬 (URDF: 접근=-Z, 벌림=+X, 법선=-Y)
        R_grasp_to_tcp = R.from_matrix([[0, 1, 0],
                                        [0, 0,-1],
                                        [-1,0, 0]])
        R_w_tcp = R_w_grasp * R_grasp_to_tcp

        score = _score_pose(t_w, R_w_tcp, object_pos_w)
        item = dict(Pname=name, use_link_to_opt=use_link_to_opt, P=P, t_w=t_w, R_w_tcp=R_w_tcp, score=score)
        if (best is None) or (score < best["score"]):
            best = item
    return best

def auto_calibrate_frames(model_output, cam_link_pose_w, object_pos_w) -> FrameCfg:
    bestA = try_calib_combo(model_output, cam_link_pose_w, object_pos_w, use_link_to_opt=False)
    bestB = try_calib_combo(model_output, cam_link_pose_w, object_pos_w, use_link_to_opt=True)
    best = bestA if bestA["score"] < bestB["score"] else bestB
    print(f"[Auto-FrameCalib] best P={best['Pname']}, link->opt={best['use_link_to_opt']}, score={best['score']:.4f}")
    return FrameCfg(name=best["Pname"], P=best["P"], use_link_to_optical=best["use_link_to_opt"])

def get_grasp_pose_in_world_calibrated(
    model_output,
    cam_link_pose_w,              # [pos(3), quat_wxyz(4)]
    frame_cfg: FrameCfg
):
    # 1) 모델 -> 카메라
    t_m = np.asarray(model_output.translation)
    R_m = R.from_matrix(np.asarray(model_output.rotation_matrix))
    P = frame_cfg.P
    P_R = R.from_matrix(P)

    t_c = P.dot(t_m)
    R_c_grasp = P_R * R_m

    # 2) 카메라 -> 월드
    cam_pos_w = np.asarray(cam_link_pose_w[0])
    qw, qx, qy, qz = cam_link_pose_w[1]   # wxyz
    R_w_link = R.from_quat([qx, qy, qz, qw])
    R_w_cam = R_w_link * (_link_to_optical_rot() if frame_cfg.use_link_to_optical else R.identity())

    t_w_grasp = cam_pos_w + R_w_cam.apply(t_c)
    R_w_grasp = R_w_cam * R_c_grasp

    # 3) TCP 정렬 (URDF: 접근=-Z, 벌림=+X, 법선=-Y)
    R_grasp_to_tcp = R.from_matrix([[0, 1, 0],
                                    [0, 0,-1],
                                    [-1,0, 0]])
    R_w_tcp = R_w_grasp * R_grasp_to_tcp

    # 4) 반환: pos(3) + quat(WXYZ)  ← 액션 버퍼에 바로 사용 가능
    quat_xyzw = R_w_tcp.as_quat()                   # [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
    pose_wxyz = np.concatenate((t_w_grasp, quat_wxyz))
    return pose_wxyz
# ========= END PATCH 1 =========
