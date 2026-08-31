"""
transform_player_coordinates.py

Basketball Tactics MVP V1
=========================

Purpose
-------
Transform player coordinates from the broadcast camera into
real basketball court coordinates.

Input
-----
outputs/player_coordinates_with_teams.csv

This file must already contain team assignments.

Output
------
outputs/player_coordinates_with_teams.csv

The script preserves all existing columns and adds:

    court_x
    court_y

Court coordinate system
-----------------------
X: 0 -> 94 feet
Y: 0 -> 50 feet

Pipeline
--------
1. Load team-assigned player coordinates
2. Read the first video frame
3. Detect basketball court landmarks with Roboflow
4. Match detected landmarks to COURT_POINTS
5. Convert COURT_POINTS template pixels -> 94 x 50 feet
6. Calculate homography
7. Transform player positions
8. Preserve team information
9. Save output CSV

Important
---------
We do NOT clip coordinates with np.clip().

Clipping would hide calibration errors and create artificial
concentrations at the court boundaries.
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

ROOT = Path(__file__).resolve().parent


# Input after team assignment
INPUT_CSV = (
    ROOT
    / "outputs"
    / "player_coordinates_with_teams.csv"
)


# Output is intentionally the same file.
# Existing team columns are preserved.
OUTPUT_CSV = (
    ROOT
    / "outputs"
    / "player_coordinates_with_teams.csv"
)


# Prefer normalized video
VIDEO_PATH = (
    ROOT
    / "data"
    / "processed"
    / "video_30fps.mp4"
)


# Fallback video
FALLBACK_VIDEO_PATH = (
    ROOT
    / "data"
    / "videos"
    / "game_30fps.mp4"
)


# Roboflow court model
COURT_MODEL_ID = (
    "basketball-court-detection-2/22"
)


# Minimum keypoint confidence
CONFIDENCE_THRESHOLD = 0.50


# RANSAC threshold in template/feet coordinates
RANSAC_THRESHOLD = 2.0


# Real basketball court dimensions
COURT_WIDTH = 94.0
COURT_HEIGHT = 50.0


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "ROBOFLOW_API_KEY"
)

if not API_KEY:

    raise RuntimeError(
        "ROBOFLOW_API_KEY not found in .env\n"
        "Add ROBOFLOW_API_KEY to your .env file."
    )


# ============================================================
# ROBOFLOW CLIENT
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)


# ============================================================
# FIND VIDEO
# ============================================================

if VIDEO_PATH.exists():

    video_path = VIDEO_PATH

elif FALLBACK_VIDEO_PATH.exists():

    video_path = FALLBACK_VIDEO_PATH

else:

    raise FileNotFoundError(
        "Could not find input video.\n\n"
        f"Tried:\n"
        f"  {VIDEO_PATH}\n"
        f"  {FALLBACK_VIDEO_PATH}"
    )


# ============================================================
# LOAD PLAYER DATA
# ============================================================

print("=" * 70)
print("🏀 PLAYER COURT COORDINATE TRANSFORMATION")
print("=" * 70)

print()
print("Loading team-assigned player coordinates...")

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}\n\n"
        "Run team_assignment.py first."
    )


df = pd.read_csv(
    INPUT_CSV
)


original_row_count = len(df)


print(
    f"Player rows: {original_row_count}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "frame",
    "player_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "team",
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise RuntimeError(
        "Input CSV is missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in sorted(
                missing_columns
            )
        )
        + "\n\n"
        "Run team_assignment.py first."
    )


# ============================================================
# REMOVE OLD COURT COORDINATES
# ============================================================
#
# If this script is run repeatedly, remove the previous
# transformation before calculating the new one.
#
# This prevents stale court coordinates from surviving.
# ============================================================

for column in [
    "court_x",
    "court_y",
]:

    if column in df.columns:

        df = df.drop(
            columns=[column]
        )


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(
    str(video_path)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{video_path}"
    )


success, frame = cap.read()

cap.release()


if not success or frame is None:

    raise RuntimeError(
        "Could not read first frame from video."
    )


frame_height, frame_width = (
    frame.shape[:2]
)


print(
    f"Video size: "
    f"{frame_width} x {frame_height}"
)


# ============================================================
# COURT DETECTION
# ============================================================

print()
print("Detecting court landmarks...")


try:

    result = client.infer(
        frame,
        model_id=COURT_MODEL_ID,
    )

except Exception as exc:

    raise RuntimeError(
        "Roboflow court detection failed.\n\n"
        f"{exc}"
    )


predictions = result.get(
    "predictions",
    []
)


# ============================================================
# FIND COURT DETECTION
# ============================================================

courts = [
    prediction
    for prediction in predictions
    if prediction.get("class") == "court"
]


if not courts:

    raise RuntimeError(
        "No basketball court detection found."
    )


best_court = max(
    courts,
    key=lambda prediction: float(
        prediction.get(
            "confidence",
            0
        )
    ),
)


court_confidence = float(
    best_court.get(
        "confidence",
        0
    )
)


print(
    f"Court confidence: "
    f"{court_confidence:.3f}"
)


# ============================================================
# GET COURT KEYPOINTS
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
template_points_pixels = []

used_ids = []
used_confidences = []


print()
print("=" * 70)
print("📐 MATCHING COURT LANDMARKS")
print("=" * 70)


for kp in keypoints:

    # --------------------------------------------------------
    # Landmark ID
    # --------------------------------------------------------

    try:

        landmark_id = int(
            kp.get("class")
        )

    except (
        TypeError,
        ValueError,
    ):

        continue


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        kp.get(
            "confidence",
            0
        )
    )


    if confidence < CONFIDENCE_THRESHOLD:

        continue


    # --------------------------------------------------------
    # Check template
    # --------------------------------------------------------

    if landmark_id not in COURT_POINTS:

        print(
            f"⚠️ Landmark {landmark_id} "
            f"not found in COURT_POINTS"
        )

        continue


    # --------------------------------------------------------
    # Camera coordinates
    # --------------------------------------------------------

    image_x = kp.get(
        "x"
    )

    image_y = kp.get(
        "y"
    )


    if (
        image_x is None
        or image_y is None
    ):

        continue


    image_x = float(
        image_x
    )

    image_y = float(
        image_y
    )


    # --------------------------------------------------------
    # Template coordinates
    # --------------------------------------------------------

    template_x, template_y = (
        COURT_POINTS[
            landmark_id
        ]
    )


    template_x = float(
        template_x
    )

    template_y = float(
        template_y
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    image_points.append(
        [
            image_x,
            image_y,
        ]
    )


    template_points_pixels.append(
        [
            template_x,
            template_y,
        ]
    )


    used_ids.append(
        landmark_id
    )


    used_confidences.append(
        confidence
    )


    print(
        f"ID {landmark_id:2d} | "
        f"camera=({image_x:8.1f}, {image_y:8.1f}) | "
        f"template=({template_x:8.2f}, {template_y:8.2f}) | "
        f"conf={confidence:.3f}"
    )


# ============================================================
# REMOVE DUPLICATE LANDMARK IDS
# ============================================================

unique_indices = []
seen_ids = set()


for index, landmark_id in enumerate(
    used_ids
):

    if landmark_id in seen_ids:

        continue

    seen_ids.add(
        landmark_id
    )

    unique_indices.append(
        index
    )


image_points = np.asarray(
    [
        image_points[index]
        for index in unique_indices
    ],
    dtype=np.float32,
)


template_points_pixels = np.asarray(
    [
        template_points_pixels[index]
        for index in unique_indices
    ],
    dtype=np.float32,
)


used_ids = [
    used_ids[index]
    for index in unique_indices
]


print()
print(
    f"Usable unique landmarks: "
    f"{len(image_points)}"
)


if len(image_points) < 4:

    raise RuntimeError(
        "Fewer than 4 usable court landmarks."
    )


# ============================================================
# TEMPLATE CALIBRATION
# ============================================================
#
# COURT_POINTS is NOT a 94 x 50 coordinate system.
#
# It is a template/image coordinate system.
#
# Example:
#
#     x ≈ 22 → 628
#     y ≈ 210 → 584
#
# We convert that template into:
#
#     X = 0 → 94 feet
#     Y = 0 → 50 feet
#
# BEFORE calculating the homography.
# ============================================================

all_template_points = np.asarray(
    list(
        COURT_POINTS.values()
    ),
    dtype=np.float32,
)


template_min_x = float(
    np.min(
        all_template_points[:, 0]
    )
)

template_max_x = float(
    np.max(
        all_template_points[:, 0]
    )
)

template_min_y = float(
    np.min(
        all_template_points[:, 1]
    )
)

template_max_y = float(
    np.max(
        all_template_points[:, 1]
    )
)


print()
print("=" * 70)
print("📏 CALIBRATING COURT TEMPLATE")
print("=" * 70)

print()

print(
    f"Template X: "
    f"{template_min_x:.2f}"
    f" → "
    f"{template_max_x:.2f}"
)

print(
    f"Template Y: "
    f"{template_min_y:.2f}"
    f" → "
    f"{template_max_y:.2f}"
)


# ============================================================
# CHECK TEMPLATE DIMENSIONS
# ============================================================

template_width = (
    template_max_x
    - template_min_x
)

template_height = (
    template_max_y
    - template_min_y
)


if template_width <= 0:

    raise RuntimeError(
        "Invalid COURT_POINTS X range."
    )


if template_height <= 0:

    raise RuntimeError(
        "Invalid COURT_POINTS Y range."
    )


# ============================================================
# CONVERT TEMPLATE PIXELS TO COURT FEET
# ============================================================

template_points_feet = (
    np.zeros_like(
        template_points_pixels,
        dtype=np.float32,
    )
)


template_points_feet[:, 0] = (
    (
        template_points_pixels[:, 0]
        - template_min_x
    )
    / template_width
    * COURT_WIDTH
)


template_points_feet[:, 1] = (
    (
        template_points_pixels[:, 1]
        - template_min_y
    )
    / template_height
    * COURT_HEIGHT
)


print()
print("Template → court calibration:")
print(
    f"X: "
    f"{template_min_x:.2f} → "
    f"{template_max_x:.2f}"
    f" becomes "
    f"0 → {COURT_WIDTH:.1f} ft"
)

print(
    f"Y: "
    f"{template_min_y:.2f} → "
    f"{template_max_y:.2f}"
    f" becomes "
    f"0 → {COURT_HEIGHT:.1f} ft"
)


# ============================================================
# HOMOGRAPHY
# ============================================================

print()
print("=" * 70)
print("📐 CALCULATING HOMOGRAPHY")
print("=" * 70)


H, mask = cv2.findHomography(
    image_points,
    template_points_feet,
    cv2.RANSAC,
    RANSAC_THRESHOLD,
)


if H is None:

    raise RuntimeError(
        "Homography calculation failed."
    )


if mask is None:

    mask = np.ones(
        len(image_points),
        dtype=np.uint8,
    )


mask = (
    mask.ravel()
    .astype(bool)
)


inliers = int(
    mask.sum()
)


print()
print("H =")
print(H)

print()

print(
    f"Inliers: "
    f"{inliers}/{len(image_points)}"
)


# ============================================================
# HOMOGRAPHY QUALITY
# ============================================================

projected_landmarks = (
    cv2.perspectiveTransform(
        image_points.reshape(
            -1,
            1,
            2,
        ),
        H,
    )
    .reshape(
        -1,
        2,
    )
)


reprojection_errors = (
    np.linalg.norm(
        projected_landmarks
        - template_points_feet,
        axis=1,
    )
)


if mask.any():

    inlier_errors = (
        reprojection_errors[
            mask
        ]
    )

else:

    inlier_errors = (
        reprojection_errors
    )


mean_error = float(
    np.mean(
        inlier_errors
    )
)

median_error = float(
    np.median(
        inlier_errors
    )
)

max_error = float(
    np.max(
        inlier_errors
    )
)


print()
print("Homography quality:")
print(
    f"Mean error:   "
    f"{mean_error:.2f} ft"
)

print(
    f"Median error: "
    f"{median_error:.2f} ft"
)

print(
    f"Max error:    "
    f"{max_error:.2f} ft"
)


# ============================================================
# LANDMARK ERROR TABLE
# ============================================================

print()
print("Landmark reprojection errors:")

for landmark_id, error, is_inlier in zip(
    used_ids,
    reprojection_errors,
    mask,
):

    status = (
        "INLIER"
        if is_inlier
        else "OUTLIER"
    )

    print(
        f"ID {landmark_id:2d} | "
        f"{error:7.2f} ft | "
        f"{status}"
    )


# ============================================================
# TRANSFORM PLAYER POSITIONS
# ============================================================

print()
print("=" * 70)
print("🎯 TRANSFORMING PLAYER POSITIONS")
print("=" * 70)


player_points = df[
    [
        "center_x",
        "center_y",
    ]
].to_numpy(
    dtype=np.float32
)


transformed_players = (
    cv2.perspectiveTransform(
        player_points.reshape(
            -1,
            1,
            2,
        ),
        H,
    )
    .reshape(
        -1,
        2,
    )
)


df["court_x"] = (
    transformed_players[:, 0]
)

df["court_y"] = (
    transformed_players[:, 1]
)


# ============================================================
# RAW TRANSFORMED RANGE
# ============================================================

print()
print("=" * 70)
print("🔎 RAW TRANSFORMED RANGE")
print("=" * 70)

print()

print(
    f"X: "
    f"{df['court_x'].min():.2f}"
    f" → "
    f"{df['court_x'].max():.2f}"
)

print(
    f"Y: "
    f"{df['court_y'].min():.2f}"
    f" → "
    f"{df['court_y'].max():.2f}"
)


# ============================================================
# REMOVE NON-FINITE VALUES
# ============================================================

finite_mask = (
    np.isfinite(
        df["court_x"]
    )
    &
    np.isfinite(
        df["court_y"]
    )
)


nonfinite_count = int(
    (~finite_mask).sum()
)


df = df[
    finite_mask
].copy()


# ============================================================
# COURT BOUNDARY DIAGNOSTICS
# ============================================================

inside_mask = (
    (df["court_x"] >= 0.0)
    &
    (df["court_x"] <= COURT_WIDTH)
    &
    (df["court_y"] >= 0.0)
    &
    (df["court_y"] <= COURT_HEIGHT)
)


inside_count = int(
    inside_mask.sum()
)


outside_count = int(
    (~inside_mask).sum()
)


print()
print("=" * 70)
print("📊 COURT RANGE CHECK")
print("=" * 70)

print()

print(
    f"Inside 94 x 50 court: "
    f"{inside_count}"
)

print(
    f"Outside court:        "
    f"{outside_count}"
)

print(
    f"Non-finite removed:   "
    f"{nonfinite_count}"
)


# ============================================================
# IMPORTANT
# ============================================================
#
# We intentionally DO NOT do:
#
#     np.clip(court_x, 0, 94)
#     np.clip(court_y, 0, 50)
#
# Clipping would make bad points appear valid.
#
# The heatmap should use genuine court positions only.
# ============================================================


# ============================================================
# KEEP VALID COURT POSITIONS
# ============================================================
#
# For the final player-coordinate dataset, keep only positions
# that actually fall inside the physical court.
#
# This prevents camera/background detections from polluting
# the heatmap.
# ============================================================

df = df[
    inside_mask
].copy()


if len(df) == 0:

    raise RuntimeError(
        "No player positions landed inside the "
        "94 x 50 ft court.\n\n"
        "The court landmark mapping needs calibration."
    )


# ============================================================
# FINAL RANGE
# ============================================================

print()
print("=" * 70)
print("📍 FINAL COURT COORDINATE RANGE")
print("=" * 70)

print()

print(
    f"Court X: "
    f"{df['court_x'].min():.2f}"
    f" → "
    f"{df['court_x'].max():.2f}"
)

print(
    f"Court Y: "
    f"{df['court_y'].min():.2f}"
    f" → "
    f"{df['court_y'].max():.2f}"
)


# ============================================================
# TEAM SUMMARY
# ============================================================

print()
print("=" * 70)
print("👕 TEAM COORDINATE SUMMARY")
print("=" * 70)

print()

team_summary = (
    df
    .groupby("team")[
        [
            "court_x",
            "court_y",
        ]
    ]
    .agg(
        [
            "min",
            "max",
            "mean",
        ]
    )
)


print(
    team_summary.to_string()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("✅ PLAYER COURT TRANSFORMATION COMPLETE")
print("=" * 70)

print()

print(
    f"Input rows:          "
    f"{original_row_count}"
)

print(
    f"Output rows:         "
    f"{len(df)}"
)

print(
    f"Removed rows:        "
    f"{original_row_count - len(df)}"
)

print(
    f"Landmarks:           "
    f"{len(image_points)}"
)

print(
    f"Homography inliers:  "
    f"{inliers}/{len(image_points)}"
)

print(
    f"Mean reprojection:   "
    f"{mean_error:.2f} ft"
)

print(
    f"Median reprojection: "
    f"{median_error:.2f} ft"
)

print(
    f"Inside court:        "
    f"{inside_count}"
)

print(
    f"Outside court:       "
    f"{outside_count}"
)

print()

print(
    f"Output:\n"
    f"{OUTPUT_CSV}"
)

print()
print("Next step:")
print(
    "  python heatmap.py"
)

print("=" * 70)