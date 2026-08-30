import cv2
import numpy as np
import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

MODEL_ID = "basketball-court-detection-2/22"

VIDEO_PATH = "data/videos/game_30fps.mp4"

OUTPUT_PATH = "outputs/court_landmarks_test.png"


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:
    print("❌ ROBOFLOW_API_KEY is not set.")
    print("Make sure your .env file contains:")
    print()
    print("ROBOFLOW_API_KEY=your_key_here")
    exit(1)


# ============================================================
# ROBOFLOW CLIENT
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


# ============================================================
# READ FIRST FRAME FROM VIDEO
# ============================================================

print("🎥 Reading first frame...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Could not open video:")
    print(VIDEO_PATH)
    exit(1)

success, frame = cap.read()

cap.release()

if not success or frame is None:
    print(f"❌ Could not read first frame:")
    print(VIDEO_PATH)
    exit(1)

print("✅ First frame loaded")

height, width = frame.shape[:2]

print(f"📐 Frame size: {width} x {height}")


# ============================================================
# COURT INFERENCE
# ============================================================

print()
print("🚀 Detecting court landmarks...")

try:
    result = client.infer(
        frame,
        model_id=MODEL_ID
    )

except Exception as e:
    print()
    print("❌ Roboflow inference failed")
    print(e)
    exit(1)


print("✅ Court inference complete")


# ============================================================
# GET COURT PREDICTIONS
# ============================================================

predictions = result.get("predictions", [])

courts = [
    prediction
    for prediction in predictions
    if prediction.get("class") == "court"
]


if not courts:
    print("❌ No court detection found.")
    exit(1)


# ============================================================
# SELECT BEST COURT
# ============================================================

best_court = max(
    courts,
    key=lambda prediction: prediction.get("confidence", 0)
)

court_confidence = best_court.get("confidence", 0)

print()
print("===================================")
print("🏀 BEST COURT DETECTION")
print("===================================")
print(f"Confidence: {court_confidence:.3f}")
print(f"Number of court detections: {len(courts)}")


# ============================================================
# GET KEYPOINTS
# ============================================================

keypoints = best_court.get("keypoints", [])

if not keypoints:
    print("❌ Court detection contains no keypoints.")
    exit(1)


print(f"Keypoints detected: {len(keypoints)}")
print("===================================")


# ============================================================
# DRAW COURT LANDMARKS
# ============================================================

output = frame.copy()

valid_points = []

print()
print("📐 COURT LANDMARKS")
print("===================================")

for keypoint in keypoints:

    x = float(keypoint["x"])
    y = float(keypoint["y"])

    confidence = float(
        keypoint.get("confidence", 0)
    )

    class_id = keypoint.get("class_id")
    class_name = keypoint.get("class")

    print(
        f"ID {str(class_name):>3} | "
        f"x={x:7.1f} | "
        f"y={y:7.1f} | "
        f"confidence={confidence:.3f}"
    )

    # Ignore very uncertain landmarks
    if confidence < 0.50:
        continue

    # Make sure coordinates are inside the frame
    if not (0 <= x < width and 0 <= y < height):
        continue

    x_int = int(round(x))
    y_int = int(round(y))

    valid_points.append(
        (x_int, y_int)
    )

    # Draw landmark
    cv2.circle(
        output,
        (x_int, y_int),
        8,
        (0, 255, 0),
        -1
    )

    # Draw white outline
    cv2.circle(
        output,
        (x_int, y_int),
        10,
        (255, 255, 255),
        2
    )

    # Label
    label = str(class_name)

    cv2.putText(
        output,
        label,
        (x_int + 12, y_int - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW COURT BOUNDING BOX
# ============================================================

court_x = float(best_court["x"])
court_y = float(best_court["y"])
court_w = float(best_court["width"])
court_h = float(best_court["height"])

x1 = max(
    0,
    int(court_x - court_w / 2)
)

y1 = max(
    0,
    int(court_y - court_h / 2)
)

x2 = min(
    width - 1,
    int(court_x + court_w / 2)
)

y2 = min(
    height - 1,
    int(court_y + court_h / 2)
)

cv2.rectangle(
    output,
    (x1, y1),
    (x2, y2),
    (255, 255, 255),
    2
)


# ============================================================
# INFORMATION PANEL
# ============================================================

cv2.rectangle(
    output,
    (20, 20),
    (390, 105),
    (0, 0, 0),
    -1
)

cv2.putText(
    output,
    "COURT DETECTION",
    (35, 52),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    output,
    f"Confidence: {court_confidence:.3f}",
    (35, 82),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE IMAGE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

saved = cv2.imwrite(
    OUTPUT_PATH,
    output
)

if not saved:
    print()
    print("❌ Failed to save output image.")
    exit(1)


# ============================================================
# SUMMARY
# ============================================================

print()
print("===================================")
print("✅ COURT LANDMARK TEST COMPLETE")
print("===================================")
print(f"Valid landmarks: {len(valid_points)}")
print(f"Output: {OUTPUT_PATH}")
print("===================================")