import cv2
import os
import numpy as np

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# ==========================================
# READ FIRST FRAME
# ==========================================

cap = cv2.VideoCapture(
    "data/videos/game_30fps.mp4"
)

success, frame = cap.read()
cap.release()

if not success:
    print("❌ Could not read video")
    exit()


# ==========================================
# COURT DETECTION
# ==========================================

print("🚀 Detecting court...")

result = client.infer(
    frame,
    model_id="basketball-court-detection-2/22"
)

courts = result["predictions"]

if not courts:
    print("❌ No court detected")
    exit()


# Pick strongest court detection
court = max(
    courts,
    key=lambda x: x["confidence"]
)

print(
    f"✅ Court confidence: "
    f"{court['confidence']:.3f}"
)


# ==========================================
# DRAW LANDMARKS
# ==========================================

points = []

for kp in court["keypoints"]:

    confidence = kp["confidence"]

    if confidence < 0.5:
        continue

    x = int(kp["x"])
    y = int(kp["y"])

    points.append((x, y))

    # Draw landmark
    cv2.circle(
        frame,
        (x, y),
        8,
        (0, 255, 255),
        -1
    )

    # Label
    cv2.putText(
        frame,
        str(kp["class"]),
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# ==========================================
# DRAW CONNECTIONS
# ==========================================

# Sort points roughly by x position
# so we can visually inspect the geometry.

points_sorted = sorted(
    points,
    key=lambda p: p[0]
)

for i in range(
    len(points_sorted) - 1
):

    p1 = points_sorted[i]
    p2 = points_sorted[i + 1]

    cv2.line(
        frame,
        p1,
        p2,
        (255, 0, 255),
        2
    )


# ==========================================
# SAVE
# ==========================================

output_path = (
    "outputs/court_geometry.jpg"
)

cv2.imwrite(
    output_path,
    frame
)

print()
print("===================================")
print("✅ COURT GEOMETRY SAVED")
print("===================================")
print(
    f"Output: {output_path}"
)
print(
    f"Landmarks used: {len(points)}"
)
print("===================================")
