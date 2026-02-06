import pyrealsense2 as rs
import numpy as np
import cv2
import os

def main():
    # Create a context object. This object owns the handles to all connected realsense devices
    ctx = rs.context()
    devices = ctx.query_devices()

    if not devices:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} connected RealSense device(s):")
    for i, device in enumerate(devices):
        print(f"  Device {i+1}:")
        print(f"    Name: {device.get_info(rs.camera_info.name)}")
        print(f"    Serial Number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"    Product Line: {device.get_info(rs.camera_info.product_line)}")

    # Output directory
    output_dir = "/home/hyunjin/RBY1_migration/rby1_ws/rby1-data-collection/tools/camera_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAttempting to capture one frame from each device into '{output_dir}'...")

    # Iterate over all connected devices (process sequentially to avoid USB bandwidth conflicts)
    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"\nProcessing device: {serial_number}")

        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_started = False

        try:
            # Enable the specific device
            config.enable_device(serial_number)

            # Check for RGB sensor on the device
            found_rgb = False
            for s in device.sensors:
                print(f"  Available sensor: {s.get_info(rs.camera_info.name)}")
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            
            if not found_rgb:
                print(f"  - No RGB camera found on device {serial_number}. Skipping.")
                continue

            # Configure and start the pipeline for the current device
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            try:
                pipeline.start(config)
                pipeline_started = True
            except Exception as e:
                print(f"  - Failed to start pipeline for device {serial_number}: {e}. Skipping.")
                continue

            # Create an align object
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Wait for a coherent pair of frames (increased timeout to 10 seconds)
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000) # 5-second timeout
            except RuntimeError as e:
                print(f"  - Could not get frame from device {serial_number}: {e}. Skipping.")
                continue
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print(f"  - Failed to get aligned frames from device {serial_number}. Skipping.")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Define unique filenames using the serial number
            color_filename = os.path.join(output_dir, f"color_frame_{serial_number}.png")
            depth_filename = os.path.join(output_dir, f"depth_frame_{serial_number}.png")

            # Save the images
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_colormap)
            print(f"  - Successfully saved images to {color_filename} and {depth_filename}")

        except Exception as e:
            print(f"  - An unexpected error occurred with device {serial_number}: {e}")
        finally:
            # Stop the pipeline only if it was started
            if pipeline_started:
                try:
                    print(f"  - Stopping pipeline for device {serial_number}.")
                    pipeline.stop()
                except Exception as e:
                    print(f"  - Error stopping pipeline: {e}")

    print("\nAll devices processed.")

if __name__ == '__main__':
    main()
