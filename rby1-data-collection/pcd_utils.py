import numpy as np
import struct


def downsample_pointcloud(points, colors, target_size=2048):
    """
    Downsample point cloud to target size using random sampling.
    
    Args:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255)
        target_size: Target number of points (default 2048)
    
    Returns:
        downsampled_points: Mx3 array of XYZ coordinates (M <= target_size)
        downsampled_colors: Mx3 array of RGB colors (0-255)
    """
    n_points = len(points)
    
    if n_points <= target_size:
        # Already smaller than target, return as is
        return points, colors
    
    # Randomly sample indices
    indices = np.random.choice(n_points, target_size, replace=False)
    
    return points[indices], colors[indices]


def rgbd_to_pointcloud(rgb, depth, fx, fy, cx, cy, depth_scale=1000.0, max_points=None):
    """
    Convert RGB-D images to point cloud.
    
    Args:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth image (H, W) uint16, in millimeters
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates
        depth_scale: Scale to convert depth to meters (default 1000 for mm to m)
        max_points: Maximum number of points to return (downsamples if needed)
    
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255)
    """
    H, W = depth.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Valid depth mask (non-zero)
    valid = depth > 0
    
    # Convert depth to meters
    z = depth[valid].astype(np.float32) / depth_scale
    
    # Back-project to 3D
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    # Stack to get points
    points = np.stack([x, y, z], axis=-1)
    
    # Get corresponding colors
    colors = rgb[valid]
    
    # Downsample if requested
    if max_points is not None and len(points) > max_points:
        points, colors = downsample_pointcloud(points, colors, max_points)
    
    return points, colors


def save_pcd(filename, points, colors=None):
    """
    Save point cloud to PCD (Point Cloud Data) file format.
    
    Args:
        filename: Output .pcd file path
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255), optional
    """
    n_points = len(points)
    has_color = colors is not None
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        
        if has_color:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
        else:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        
        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        
        # Write points
        if has_color:
            for i in range(n_points):
                x, y, z = points[i]
                r, g, b = colors[i]
                # Pack RGB into single float32
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed}\n")
        else:
            for i in range(n_points):
                x, y, z = points[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_pcd_binary(filename, points, colors=None):
    """
    Save point cloud to binary PCD format (faster and smaller).
    
    Args:
        filename: Output .pcd file path
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255), optional
    """
    n_points = len(points)
    has_color = colors is not None
    
    with open(filename, 'wb') as f:
        # Write ASCII header
        header = "# .PCD v0.7 - Point Cloud Data file format\n"
        header += "VERSION 0.7\n"
        
        if has_color:
            header += "FIELDS x y z rgb\n"
            header += "SIZE 4 4 4 4\n"
            header += "TYPE F F F U\n"
            header += "COUNT 1 1 1 1\n"
        else:
            header += "FIELDS x y z\n"
            header += "SIZE 4 4 4\n"
            header += "TYPE F F F\n"
            header += "COUNT 1 1 1\n"
        
        header += f"WIDTH {n_points}\n"
        header += "HEIGHT 1\n"
        header += "VIEWPOINT 0 0 0 1 0 0 0\n"
        header += f"POINTS {n_points}\n"
        header += "DATA binary\n"
        
        f.write(header.encode('ascii'))
        
        # Write binary data
        if has_color:
            for i in range(n_points):
                x, y, z = points[i]
                r, g, b = colors[i]
                # Pack RGB into single uint32
                rgb_packed = struct.pack('BBBB', b, g, r, 0)
                f.write(struct.pack('fff', x, y, z))
                f.write(rgb_packed)
        else:
            for i in range(n_points):
                x, y, z = points[i]
                f.write(struct.pack('fff', x, y, z))


# RealSense D435 camera intrinsics (default values)
# These should be adjusted based on actual camera calibration
REALSENSE_D435_INTRINSICS = {
    'fx': 615.0,  # Focal length X
    'fy': 615.0,  # Focal length Y
    'cx': 320.0,  # Principal point X (width/2)
    'cy': 240.0,  # Principal point Y (height/2)
    'width': 640,
    'height': 480,
}

REALSENSE_D435_INTRINSICS_848x480 = {
    'fx': 615.0 * (848.0 / 640.0),
    'fy': 615.0,
    'cx': 424.0,
    'cy': 240.0,
    'width': 848,
    'height': 480,
}
