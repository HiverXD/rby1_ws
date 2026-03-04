# Point Cloud Data (PCD) 저장 기능

RGB-D 데이터를 3D Point Cloud로 변환하여 PCD 파일로 저장합니다.

## 📁 파일 구조

```
rby1-data-collection/
├── pcd_utils.py           # PCD 변환 유틸리티
├── h5_writer.py           # 자동 PCD 저장 포함 (multi-camera H5 writer)
├── main.py                # 데이터 수집
└── data/
    ├── demo_0.h5          # HDF5 데이터
    └── demo_0_pcd/        # 자동 생성된 PCD 파일들
        ├── frame_000000.pcd
        ├── frame_000001.pcd
        └── ...
```

## 🚀 사용 방법

### 1. 자동 PCD 저장 (권장)

데이터 수집 시 자동으로 PCD 파일이 생성됩니다:

```bash
python main.py --local_ip <YOUR_IP> --meta_quest_ip <QUEST_IP>
```

- HDF5 파일과 함께 `{demo_name}_pcd/` 디렉토리가 자동 생성됩니다
- 각 프레임이 `frame_XXXXXX.pcd` 형식으로 저장됩니다
- Binary PCD 형식으로 저장 (빠르고 작은 용량)

### 2. 수동 변환 (노트북)

기존 HDF5 파일을 PCD로 변환:

```python
from pcd_utils import rgbd_to_pointcloud, save_pcd_binary
from pcd_utils import REALSENSE_D435_INTRINSICS

# RGB와 Depth 이미지를 Point Cloud로 변환
points, colors = rgbd_to_pointcloud(
    rgb_image,      # (H, W, 3) uint8
    depth_image,    # (H, W) uint16 in mm
    fx=615.0,       # focal length X
    fy=615.0,       # focal length Y
    cx=320.0,       # principal point X
    cy=240.0        # principal point Y
)

# PCD 파일로 저장
save_pcd_binary('output.pcd', points, colors)
```

## 📊 PCD 파일 형식

### Binary PCD (기본)
- **장점**: 빠른 읽기/쓰기, 작은 파일 크기
- **포맷**: `.pcd` binary format
- **필드**: `x y z rgb`
- **타입**: Float32 + Uint32(RGB)

### ASCII PCD (옵션)
```python
from pcd_utils import save_pcd
save_pcd('output.pcd', points, colors)
```

## 🎯 Camera Intrinsics

### RealSense D435 기본값

**640x480 해상도:**
```python
fx = 615.0
fy = 615.0
cx = 320.0
cy = 240.0
```

**848x480 해상도:**
```python
fx = 807.5  # 615.0 * (848/640)
fy = 615.0
cx = 424.0
cy = 240.0
```

### Intrinsics 가져오기

실제 카메라 파라미터 확인:
```bash
ros2 topic echo /camera/camera/color/camera_info --once
```

또는 RealSense SDK:
```bash
rs-enumerate-devices -c
```

## 📈 Point Cloud 시각화

### Open3D 사용

```bash
# Open3D 설치
pip install open3d
```

```python
import open3d as o3d

# PCD 파일 로드
pcd = o3d.io.read_point_cloud("frame_000000.pcd")

# 시각화
o3d.visualization.draw_geometries([pcd])
```

### CloudCompare 사용

무료 Point Cloud 뷰어:
```bash
# Ubuntu
sudo snap install cloudcompare

# 실행
cloudcompare frame_000000.pcd
```

### PCL Viewer 사용

```bash
# PCL 설치
sudo apt install pcl-tools

# 실행
pcl_viewer frame_000000.pcd
```

## 🔧 고급 설정

### PCD 저장 비활성화

```python
h5_writer = H5Writer(path=output_path, save_pcd=False)
```

### 커스텀 Intrinsics

```python
from pcd_utils import rgbd_to_pointcloud

points, colors = rgbd_to_pointcloud(
    rgb, depth,
    fx=your_fx,
    fy=your_fy,
    cx=your_cx,
    cy=your_cy,
    depth_scale=1000.0  # mm to meters
)
```

## 📝 데이터 형식

### Input (RGB-D)
- **RGB**: `(H, W, 3)` uint8, 0-255
- **Depth**: `(H, W)` uint16, millimeters
- **0 값**: Invalid depth (측정 실패)

### Output (Point Cloud)
- **Points**: `(N, 3)` float32, meters
  - X: 좌우 (우측이 양수)
  - Y: 상하 (아래가 양수)
  - Z: 전후 (앞이 양수)
- **Colors**: `(N, 3)` uint8, 0-255 RGB

## 🎨 활용 예시

### 1. 3D 재구성
```python
# 여러 프레임 병합하여 3D 모델 생성
import open3d as o3d

# 모든 PCD 로드
pcds = []
for pcd_file in sorted(pcd_dir.glob('*.pcd')):
    pcds.append(o3d.io.read_point_cloud(str(pcd_file)))

# 병합
merged = pcds[0]
for pcd in pcds[1:]:
    merged += pcd

# 저장
o3d.io.write_point_cloud("merged.pcd", merged)
```

### 2. Point Cloud 필터링
```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("frame_000000.pcd")

# Statistical outlier removal
pcd_filtered = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

# Voxel downsampling
pcd_down = pcd_filtered.voxel_down_sample(voxel_size=0.01)
```

### 3. 법선 추정
```python
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
```

## 💡 팁

1. **대용량 데이터**: Binary PCD 사용 (ASCII보다 10배 이상 빠름)
2. **메모리 절약**: Voxel downsampling으로 point 수 줄이기
3. **정확한 Intrinsics**: 실제 카메라 파라미터 사용 권장
4. **Invalid Depth**: 0 값은 자동으로 필터링됨

## 🐛 문제 해결

### "ImportError: pcd_utils"
```bash
# 현재 디렉토리에서 실행하거나
cd /home/nvidia/rby1_ws/rby1-data-collection

# 또는 PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:/home/nvidia/rby1_ws/rby1-data-collection
```

### 빈 Point Cloud
- Depth 이미지가 모두 0인지 확인
- Camera intrinsics가 올바른지 확인
- Depth scale 확인 (mm vs m)

### 왜곡된 Point Cloud
- Intrinsics (fx, fy, cx, cy) 확인
- RGB와 Depth가 정렬되어 있는지 확인
- RealSense의 경우 `image_rect_raw` 사용 권장

## 📚 참고 자료

- [PCD File Format](http://pointclouds.org/documentation/tutorials/pcd_file_format.html)
- [Open3D Documentation](http://www.open3d.org/docs/release/)
- [RealSense D435 Specs](https://www.intelrealsense.com/depth-camera-d435/)
