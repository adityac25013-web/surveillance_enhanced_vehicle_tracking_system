"""
Quick test: can OpenCV read from your webcam?
Run from the backend folder:  python test_cam.py
"""
import cv2

print("Testing camera indices 0-3 with DirectShow (CAP_DSHOW)...\n")

for idx in range(0, 4):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok_open = cap.isOpened()
    print(f"Index {idx} (DSHOW): opened={ok_open}")
    if ok_open:
        ret, frame = cap.read()
        print(f"  read ok={ret}, shape={frame.shape if ret else None}")
    cap.release()
    print()

print("If all show opened=False or read ok=False, try:")
print("  - Windows Settings > Privacy > Camera: allow desktop apps")
print("  - Close other apps using the camera (Zoom, Teams, etc.)")
