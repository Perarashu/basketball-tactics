import cv2
import os

video_path = "data/videos/game.mp4"
output_path = "data/frame.jpg"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

# Jump to 5 seconds into the video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)

success, frame = cap.read()

if not success:
    print("❌ Could not read frame")
    cap.release()
    exit()

os.makedirs("data", exist_ok=True)

cv2.imwrite(output_path, frame)

print(f"✅ Frame saved to {output_path}")

cap.release()