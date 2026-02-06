#!/usr/bin/env python3
"""
Advanced RealSense Camera Diagnostic Tool
Provides detailed information about camera connectivity, power, and USB status
"""

import pyrealsense2 as rs
import time

def check_usb_info():
    """Check USB connection info for each device"""
    print("=" * 80)
    print("USB DEVICE INFORMATION")
    print("=" * 80)
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    for i, device in enumerate(devices):
        print(f"\n[Device {i+1}]")
        print(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Name: {device.get_info(rs.camera_info.name)}")
        print(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        print(f"  USB Port ID: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        
        # Check if device is connected
        if device.is_connected():
            print(f"  Status: ✓ Connected")
        else:
            print(f"  Status: ✗ NOT Connected")
        
        # Check sensors
        print(f"  Sensors:")
        for sensor in device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"    - {sensor_name}")


def test_individual_camera(serial_number, timeout_ms=15000):
    """Test a single camera with extended timeout"""
    print(f"\n{'='*80}")
    print(f"TESTING CAMERA: {serial_number}")
    print(f"{'='*80}")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # Configure for this specific device
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print(f"[1/4] Starting pipeline... ", end="", flush=True)
        profile = pipeline.start(config)
        print("✓ OK")
        
        # Get device info
        device = profile.get_device()
        print(f"[2/4] Device info:")
        print(f"      USB Port: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        
        # Wait for frames with extended timeout
        print(f"[3/4] Waiting for frames (timeout: {timeout_ms}ms)... ", end="", flush=True)
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        print("✓ OK")
        
        # Check frame info
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame and depth_frame:
            print(f"[4/4] Frame validation:")
            print(f"      Color frame: {color_frame.get_width()}x{color_frame.get_height()}")
            print(f"      Depth frame: {depth_frame.get_width()}x{depth_frame.get_height()}")
            print(f"  Result: ✓ SUCCESS - Camera working correctly")
            return True
        else:
            print(f"[4/4] Frame validation: ✗ FAILED - No frames received")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    finally:
        try:
            pipeline.stop()
        except:
            pass


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "RealSense Camera Diagnostic Tool" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    # Step 1: Check USB info
    check_usb_info()
    
    # Step 2: Test each camera individually
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print("\n\n" + "="*80)
    print("INDIVIDUAL CAMERA TESTS")
    print("="*80)
    
    results = {}
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"\n[{i+1}/{len(devices)}]", end=" ")
        
        # Add delay between camera tests to avoid USB conflicts
        if i > 0:
            print("(waiting 3 seconds for USB to stabilize...)")
            time.sleep(3)
        
        success = test_individual_camera(serial, timeout_ms=15000)
        results[serial] = success
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nWorking cameras: {working}/{total}")
    for serial, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {serial}: {status}")
    
    if working < total:
        print(f"\n⚠️  RECOMMENDATIONS:")
        print(f"  1. Check USB hub power supply (need 1A per camera minimum)")
        print(f"  2. Try different USB ports on computer (not hub)")
        print(f"  3. Check cable connections - reseat cables firmly")
        print(f"  4. Try one camera at a time in different USB ports")
        print(f"  5. Update RealSense firmware: https://github.com/IntelRealSense/librealsense/wiki/Firmware")
    
    print("\n")

if __name__ == '__main__':
    main()
