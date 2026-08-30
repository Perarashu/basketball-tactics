"""
transform_player_coordinates.py

MVP production step:
    Camera player coordinates
        ↓
    Court landmark detection
        ↓
    Homography
        ↓
    640x640 court coordinates

Input:
    outputs/player_coordinates_v2.csv

Output:
    outputs/player_coordinates_with_teams.csv

The script preserves the existing team assignment columns if they
already exist.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from court_points import COURT_POINTS


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "data/videos/game_30fps.mp4"

PLAYER_INPUT = "outputs/player_coordinates_with_teams.csv"

OUTPUT_PATH = "outputs/player_coordinates_with_teams.csv"

COURT_MODEL_ID = "basketball-court-detection-2/22"

CONFIDENCE_THRESHOLD = 0.50

TEMPLATE_WIDTH = 640
TEMPLATE_HEIGHT = 640


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
# ROBOFLOW CLIENT
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)


# ============================================================
# LOAD PLAYERS
# ============================================================

print("=" * 60)
print("🏀 PLAYER COURT COORDINATE TRANSFORMATION")
print("=" * 60)

print()
print("Loading player coordinates...")

if not Path(PLAYER_INPUT).exists():
    raise RuntimeError(
        f"Missing player CSV: {PLAYER_INPUT}"
    )

players = pd.read_csv(PLAYER_INPUT)

print(
    f"Player rows: {len(players)}"
)


required_columns = [
    "frame",
    "player_id",
    "center_x",
    "center_y",
]

missing = [
    column
    for column in required_columns
    if column not in players.columns
]

if missing:
    raise RuntimeError(
        f"Player CSV missing columns: {missing}"
    )


# ============================================================
# LOAD VIDEO
# ============================================================

print()
print("Opening video...")

if not Path(VIDEO_PATH).exists():
    raise RuntimeError(
        f"Missing video: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

success, frame = cap.read()

cap.release()

if not success or frame is None:
    raise RuntimeError(
        "Could not read first video frame"
    )

height, width = frame.shape[:2]

print(
    f"Video size: {width} x {height}"
)


# ============================================================
# COURT DETECTION
# ============================================================

print()
print("Detecting court landmarks...")

result = client.infer(
    frame,
    model_id=COURT_MODEL_ID,
)

predictions = result.get(
    "predictions",
    [],
)

courts = [
    prediction
    for prediction in predictions
    if prediction.get("class") == "court"
]

if not courts:
    raise RuntimeError(
        "No court detection found."
    )


best_court = max(
    courts,
    key=lambda prediction: prediction.get(
        "confidence",
        0,
    ),
)

court_confidence = float(
    best_court.get(
        "confidence",
        0,
    )
)

print(
    f"Court confidence: {court_confidence:.3f}"
)


# ============================================================
# GET KEYPOINTS
# ============================================================

keypoints = best_court.get(
    "keypoints",
    [],
)

print(
    f"Keypoints returned: {len(keypoints)}"
)


# ============================================================
# MATCH IMAGE → TEMPLATE POINTS
# ============================================================

image_points = []
template_points = []
used_ids = []

for kp in keypoints:

    landmark_id = int(
        kp["class"]
    )

    confidence = float(
        kp.get(
            "confidence",
            0,
        )
    )

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    if landmark_id not in COURT_POINTS:
        continue

    image_x = float(kp["x"])
    image_y = float(kp["y"])

    template_x, template_y = COURT_POINTS[
        landmark_id
    ]

    image_points.append([
        image_x,
        image_y,
    ])

    template_points.append([
        float(template_x),
        float(template_y),
    ])

    used_ids.append(
        landmark_id
    )


print()
print(
    f"Usable landmarks: {len(image_points)}"
)

if len(image_points) < 4:
    raise RuntimeError(
        "Fewer than 4 usable court landmarks. "
        "Cannot calculate homography."
    )


# ============================================================
# HOMOGRAPHY
# ============================================================

image_points = np.array(
    image_points,
    dtype=np.float32,
)

template_points = np.array(
    template_points,
    dtype=np.float32,
)

H, mask = cv2.findHomography(
    image_points,
    template_points,
    cv2.RANSAC,
    5.0,
)

if H is None:
    raise RuntimeError(
        "Homography calculation failed."
    )


inliers = int(
    mask.sum()
) if mask is not None else 0


print()
print("=" * 60)
print("📐 HOMOGRAPHY")
print("=" * 60)

print(
    f"Landmarks: {len(image_points)}"
)

print(
    f"Inliers:   {inliers}/{len(image_points)}"
)


# ============================================================
# REPROJECTION ERROR
# ============================================================

projected = cv2.perspectiveTransform(
    image_points.reshape(-1, 1, 2),
    H,
).reshape(-1, 2)

errors = np.linalg.norm(
    projected - template_points,
    axis=1,
)

mean_error = float(
    np.mean(errors)
)

max_error = float(
    np.max(errors)
)

print(
    f"Mean reprojection error: {mean_error:.2f}"
)

print(
    f"Max reprojection error:  {max_error:.2f}"
)


# ============================================================
# TRANSFORM PLAYER CENTERS
# ============================================================

print()
print("Transforming player coordinates...")

camera_points = players[
    [
        "center_x",
        "center_y",
    ]
].to_numpy(
    dtype=np.float32
)

valid_mask = np.isfinite(
    camera_points
).all(axis=1)


court_points = np.full(
    (
        len(players),
        2,
    ),
    np.nan,
    dtype=np.float32,
)


if valid_mask.any():

    transformed = cv2.perspectiveTransform(
        camera_points[valid_mask].reshape(
            -1,
            1,
            2,
        ),
        H,
    ).reshape(
        -1,
        2,
    )

    court_points[valid_mask] = transformed


players["court_x"] = court_points[:, 0]
players["court_y"] = court_points[:, 1]


# ============================================================
# PRESERVE / INITIALIZE TEAM COLUMNS
# ============================================================

if "team" not in players.columns:

    players["team"] = "UNKNOWN"

if "jersey_color" not in players.columns:

    players["jersey_color"] = "UNKNOWN"


# ============================================================
# SAVE
# ============================================================

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True,
)

players.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("📊 TRANSFORMATION COMPLETE")
print("=" * 60)

print(
    f"Player rows:      {len(players)}"
)

print(
    f"Valid positions:  {valid_mask.sum()}"
)

print(
    f"Court coordinates: {players['court_x'].notna().sum()}"
)

print()
print(
    "Output columns:"
)

print(
    players.columns.tolist()
)

print()
print(
    f"Output: {OUTPUT_PATH}"
)

print("=" * 60)
print("✅ PLAYER COURT COORDINATES READY")
print("=" * 60)