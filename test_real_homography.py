import cv2
import numpy as np
import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from court_points import COURT_POINTS


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "data/videos/game_30fps.mp4"
OUTPUT_PATH = "outputs/real_homography_test.png"

COURT_MODEL_ID = "basketball-court-detection-2/22"

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found in .env"
    )


# ============================================================
# ROBOFLOW
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


# ============================================================
# READ FIRST VIDEO FRAME
# ============================================================

print("=" * 60)
print("🏀 REAL COURT HOMOGRAPHY TEST")
print("=" * 60)

print()
print("🎥 Reading first video frame...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

success, frame = cap.read()

cap.release()

if not success or frame is None:
    raise RuntimeError(
        f"Could not read first frame from: {VIDEO_PATH}"
    )

height, width = frame.shape[:2]

print("✅ First frame loaded")

print(
    f"Frame size: {width} x {height}"
)


# ============================================================
# COURT DETECTION
# ============================================================

print()
print("🚀 Detecting court landmarks...")

result = client.infer(
    frame,
    model_id=COURT_MODEL_ID
)

predictions = result.get(
    "predictions",
    []
)

courts = [
    p for p in predictions
    if p.get("class") == "court"
]

if not courts:
    raise RuntimeError(
        "❌ No court detection found"
    )

best_court = max(
    courts,
    key=lambda p: p.get("confidence", 0)
)

print(
    f"✅ Court confidence: "
    f"{best_court.get('confidence', 0):.3f}"
)


# ============================================================
# KEYPOINTS
# ============================================================

keypoints = best_court.get(
    "keypoints",
    []
)

print(
    f"Keypoints returned: "
    f"{len(keypoints)}"
)


# ============================================================
# MATCH LANDMARKS
# ============================================================

image_points = []
template_points = []
used_ids = []

print()
print("=" * 60)
print("📐 MATCHING LANDMARKS")
print("=" * 60)

for kp in keypoints:

    landmark_id = int(
        kp["class"]
    )

    confidence = float(
        kp.get("confidence", 0)
    )

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    if landmark_id not in COURT_POINTS:
        print(
            f"⚠️ ID {landmark_id} "
            f"not present in template"
        )
        continue

    image_x = float(kp["x"])
    image_y = float(kp["y"])

    template_x, template_y = COURT_POINTS[
        landmark_id
    ]

    image_points.append([
        image_x,
        image_y
    ])

    template_points.append([
        template_x,
        template_y
    ])

    used_ids.append(
        landmark_id
    )

    print(
        f"ID {landmark_id:2d} | "
        f"camera=({image_x:7.1f}, {image_y:7.1f}) | "
        f"template=({template_x:7.1f}, {template_y:7.1f}) | "
        f"conf={confidence:.3f}"
    )


# ============================================================
# CHECK
# ============================================================

print()
print(
    f"Usable landmarks: "
    f"{len(image_points)}"
)

if len(image_points) < 4:
    raise RuntimeError(
        "❌ Fewer than 4 usable landmarks"
    )


# ============================================================
# HOMOGRAPHY
# ============================================================

image_points = np.array(
    image_points,
    dtype=np.float32
)

template_points = np.array(
    template_points,
    dtype=np.float32
)

H, mask = cv2.findHomography(
    image_points,
    template_points,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise RuntimeError(
        "❌ Homography calculation failed"
    )

inliers = int(
    mask.sum()
) if mask is not None else 0


print()
print("=" * 60)
print("✅ REAL HOMOGRAPHY CALCULATED")
print("=" * 60)

print()
print("H =")
print(H)

print()
print(
    f"Inliers: "
    f"{inliers}/{len(image_points)}"
)


# ============================================================
# REPROJECTION ERROR
# ============================================================

projected = cv2.perspectiveTransform(
    image_points.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

errors = np.linalg.norm(
    projected - template_points,
    axis=1
)

print()
print("Reprojection errors:")

for landmark_id, error in zip(
    used_ids,
    errors
):

    print(
        f"ID {landmark_id:2d}: "
        f"{error:.2f}px"
    )

print()

print(
    f"Mean error: "
    f"{errors.mean():.2f}px"
)

print(
    f"Max error: "
    f"{errors.max():.2f}px"
)


# ============================================================
# DRAW DETECTED LANDMARKS
# ============================================================

output = frame.copy()

for kp in keypoints:

    landmark_id = int(
        kp["class"]
    )

    confidence = float(
        kp.get("confidence", 0)
    )

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    x = int(kp["x"])
    y = int(kp["y"])

    cv2.circle(
        output,
        (x, y),
        7,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        output,
        str(landmark_id),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW PROJECTED TEMPLATE POINTS
# ============================================================

inverse_H = np.linalg.inv(H)

reprojected_camera = cv2.perspectiveTransform(
    template_points.reshape(-1, 1, 2),
    inverse_H
).reshape(-1, 2)

for landmark_id, point in zip(
    used_ids,
    reprojected_camera
):

    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        output,
        (x, y),
        4,
        (255, 0, 0),
        -1
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_PATH,
    output
)


print()
print("=" * 60)
print("✅ REAL HOMOGRAPHY TEST COMPLETE")
print("=" * 60)

print(
    f"Output: {OUTPUT_PATH}"
)

print("=" * 60)
