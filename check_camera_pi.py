import cv2
import glob
import os

print("==================================================")
print("Raspberry Pi Camera Diagnostic Utility")
print("==================================================")

print("\n1. Listing all /dev/video* devices:")
devices = sorted(glob.glob('/dev/video*'))
print(f"Found devices: {devices}")

for dev in devices:
    print(f"\nTrying to open path: {dev}...")
    try:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ SUCCESS: {dev} opened and read frame of shape {frame.shape}")
            else:
                print(f"  ⚠ WARNING: {dev} opened but failed to read frame (might be a metadata/sub-channel node)")
            cap.release()
        else:
            print(f"  ✗ FAILED: {dev} could not be opened")
    except Exception as e:
        print(f"  ✗ ERROR: {dev} threw exception: {e}")

print("\n2. Trying numeric camera indices 0-5:")
for i in range(6):
    print(f"Trying camera index {i}...")
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ SUCCESS: Index {i} opened and read frame of shape {frame.shape}")
            else:
                print(f"  ⚠ WARNING: Index {i} opened but failed to read frame")
            cap.release()
        else:
            print(f"  ✗ FAILED: Index {i} could not be opened")
    except Exception as e:
        print(f"  ✗ ERROR: Index {i} threw exception: {e}")

print("\n==================================================")
print("Diagnostic complete.")
print("==================================================")
