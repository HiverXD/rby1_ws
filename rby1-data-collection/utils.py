import numpy as np 
from sensor_msgs.msg import Image
import os

def rosimg_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image → np.ndarray (HxWxC, uint8)."""
    dtype = np.uint8  # adjust if msg.encoding differs
    img = np.frombuffer(msg.data, dtype=dtype)
    img = img.reshape((msg.height, msg.width, -1))  # works for RGB/BGR/mono # (H, W, C)
    return img


def get_next_h5_path(base_dir="/home/nvidia/rby1_ws/rby1-data-collection/data"):
# def get_next_h5_path(base_dir="/media/nvidia/T7/Demo"):
    # Count existing .h5 files
    existing = [f for f in os.listdir(base_dir) if f.endswith(".h5")]
    next_index = len(existing)
    # Create new path
    return os.path.join(base_dir, f"demo_{next_index}.h5")
