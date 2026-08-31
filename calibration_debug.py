import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from court_points import COURT_POINTS


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "data/videos/game_30fps.mp4"
PLAYER_CSV = "outputs/player_coordinates_with_teams.csv"

OUTPUT_PATH = "outputs/calibration_debug.png"

COURT_MODEL_ID = "basketball-court-detection-2/22"

CONFIDENCE_THRESHOLD = 0.50

# Real basketball court dimensions in feet
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0


# ============================================================
# LOAD ENVIRONMENT
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
# LOAD PLAYER DATA
# ============================================================

print("=" * 70)
print("🏀 CALIBRATION DEBUG")
print("=" * 70)

print()
print("Loading player coordinates...")

if not os.path.exists(PLAYER_CSV):
    raise FileNotFoundError(
        f"Player CSV not found: {PLAYER_CSV}"
    )

players = pd.read_csv(PLAYER_CSV)

required_columns = [
    "frame",
    "player_id",
    "center_x",
    "center_y",
    "team",
]

for column in required_columns:

    if column not in players.columns:
        raise RuntimeError(
            f"Missing required column: {column}"
        )

print(
    f"Player rows: {len(players)}"
)


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

success, frame = cap.read()

cap.release()

if not success or frame is None:
    raise RuntimeError(
        "Could not read first video frame."
    )

frame_height, frame_width = frame.shape[:2]

print(
    f"Video size: "
    f"{frame_width} x {frame_height}"
)


# ============================================================
# DETECT COURT
# ============================================================

print()
print("Detecting court landmarks...")

result = client.infer(
    frame,
    model_id=COURT_MODEL_ID,
)

predictions = result.get(
    "predictions",
    []
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
    key=lambda prediction:
    prediction.get("confidence", 0),
)

court_confidence = float(
    best_court.get("confidence", 0)
)

print(
    f"Court confidence: "
    f"{court_confidence:.3f}"
)


# ============================================================
# GET KEYPOINTS
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
# MATCH COURT LANDMARKS
# ============================================================

image_points = []
template_points = []
landmark_ids = []

print()
print("=" * 70)
print("📐 MATCHING COURT LANDMARKS")
print("=" * 70)

for kp in keypoints:

    try:
        landmark_id = int(
            kp["class"]
        )
    except Exception:
        continue

    confidence = float(
        kp.get("confidence", 0)
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

    image_points.append(
        [image_x, image_y]
    )

    template_points.append(
        [template_x, template_y]
    )

    landmark_ids.append(
        landmark_id
    )

    print(
        f"ID {landmark_id:2d} | "
        f"camera=({image_x:7.1f}, {image_y:7.1f}) | "
        f"template=({template_x:7.1f}, {template_y:7.1f}) | "
        f"confidence={confidence:.3f}"
    )


print()
print(
    f"Usable landmarks: "
    f"{len(image_points)}"
)

if len(image_points) < 4:
    raise RuntimeError(
        "Fewer than 4 usable landmarks."
    )


# ============================================================
# CALCULATE HOMOGRAPHY
# ============================================================

image_points_np = np.array(
    image_points,
    dtype=np.float32,
)

template_points_np = np.array(
    template_points,
    dtype=np.float32,
)

H, mask = cv2.findHomography(
    image_points_np,
    template_points_np,
    cv2.RANSAC,
    5.0,
)

if H is None:
    raise RuntimeError(
        "Homography calculation failed."
    )

mask = (
    mask.flatten()
    if mask is not None
    else np.ones(
        len(image_points_np),
        dtype=np.uint8,
    )
)

inliers = int(
    mask.sum()
)


# ============================================================
# REPROJECTION ERROR
# ============================================================

projected = cv2.perspectiveTransform(
    image_points_np.reshape(-1, 1, 2),
    H,
).reshape(-1, 2)

errors = np.linalg.norm(
    projected - template_points_np,
    axis=1,
)

mean_error = float(
    np.mean(errors)
)

max_error = float(
    np.max(errors)
)

print()
print("=" * 70)
print("📊 HOMOGRAPHY QUALITY")
print("=" * 70)

print(
    f"Landmarks:              {len(image_points_np)}"
)

print(
    f"Inliers:                {inliers}/{len(image_points_np)}"
)

print(
    f"Mean reprojection error: {mean_error:.2f}"
)

print(
    f"Max reprojection error:  {max_error:.2f}"
)


# ============================================================
# TRANSFORM PLAYER POSITIONS
#
# IMPORTANT:
#
# The existing CSV already contains court_x/court_y.
# This script independently transforms the ORIGINAL camera
# coordinates so we can compare the result against the saved
# coordinates.
# ============================================================

print()
print("Transforming player positions...")

camera_points = players[
    ["center_x", "center_y"]
].to_numpy(
    dtype=np.float32
)

valid_mask = np.isfinite(
    camera_points
).all(axis=1)

transformed = np.full(
    (len(players), 2),
    np.nan,
    dtype=np.float32,
)

if valid_mask.any():

    transformed_points = cv2.perspectiveTransform(
        camera_points[
            valid_mask
        ].reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    transformed[
        valid_mask
    ] = transformed_points


players["debug_template_x"] = transformed[:, 0]
players["debug_template_y"] = transformed[:, 1]


# ============================================================
# TEMPLATE → REAL COURT COORDINATES
#
# The Roboflow template is 640x640.
#
# We map its useful court bounding region to:
#
# X = 0 ... 94 feet
# Y = 0 ... 50 feet
#
# We use the actual landmark extent instead of blindly using
# 0...640. This prevents the heatmap from being compressed
# into one corner.
# ============================================================

template_array = np.array(
    list(COURT_POINTS.values()),
    dtype=np.float32,
)

template_min_x = float(
    template_array[:, 0].min()
)

template_max_x = float(
    template_array[:, 0].max()
)

template_min_y = float(
    template_array[:, 1].min()
)

template_max_y = float(
    template_array[:, 1].max()
)

print()
print("=" * 70)
print("📏 TEMPLATE CALIBRATION")
print("=" * 70)

print(
    f"Template X: "
    f"{template_min_x:.2f} → {template_max_x:.2f}"
)

print(
    f"Template Y: "
    f"{template_min_y:.2f} → {template_max_y:.2f}"
)


def template_to_court_x(x):

    return (
        (x - template_min_x)
        / (template_max_x - template_min_x)
        * COURT_LENGTH
    )


def template_to_court_y(y):

    return (
        (y - template_min_y)
        / (template_max_y - template_min_y)
        * COURT_WIDTH
    )


players["debug_court_x"] = (
    players["debug_template_x"]
    .apply(
        lambda value:
        template_to_court_x(value)
        if np.isfinite(value)
        else np.nan
    )
)

players["debug_court_y"] = (
    players["debug_template_y"]
    .apply(
        lambda value:
        template_to_court_y(value)
        if np.isfinite(value)
        else np.nan
    )
)


# ============================================================
# COURT DRAWING
# ============================================================

def draw_court(ax):

    # --------------------------------------------------------
    # Outer court
    # --------------------------------------------------------

    ax.add_patch(
        Rectangle(
            (0, 0),
            COURT_LENGTH,
            COURT_WIDTH,
            fill=False,
            linewidth=2,
        )
    )

    # --------------------------------------------------------
    # Half-court line
    # --------------------------------------------------------

    ax.plot(
        [COURT_LENGTH / 2, COURT_LENGTH / 2],
        [0, COURT_WIDTH],
        linewidth=1.5,
    )

    # --------------------------------------------------------
    # Center circle
    # --------------------------------------------------------

    ax.add_patch(
        Circle(
            (COURT_LENGTH / 2, COURT_WIDTH / 2),
            6,
            fill=False,
            linewidth=1.5,
        )
    )

    # --------------------------------------------------------
    # Paint areas
    # --------------------------------------------------------

    paint_width = 16
    paint_depth = 19

    ax.add_patch(
        Rectangle(
            (0, (COURT_WIDTH - paint_width) / 2),
            paint_depth,
            paint_width,
            fill=False,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Rectangle(
            (
                COURT_LENGTH - paint_depth,
                (COURT_WIDTH - paint_width) / 2,
            ),
            paint_depth,
            paint_width,
            fill=False,
            linewidth=1.5,
        )
    )

    # --------------------------------------------------------
    # Free throw circles
    # --------------------------------------------------------

    ax.add_patch(
        Circle(
            (paint_depth, COURT_WIDTH / 2),
            6,
            fill=False,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Circle(
            (
                COURT_LENGTH - paint_depth,
                COURT_WIDTH / 2,
            ),
            6,
            fill=False,
            linewidth=1.5,
        )
    )

    # --------------------------------------------------------
    # Baskets
    # --------------------------------------------------------

    basket_x_left = 1.25
    basket_x_right = COURT_LENGTH - 1.25
    basket_y = COURT_WIDTH / 2

    ax.add_patch(
        Circle(
            (basket_x_left, basket_y),
            0.75,
            fill=False,
            linewidth=2,
        )
    )

    ax.add_patch(
        Circle(
            (basket_x_right, basket_y),
            0.75,
            fill=False,
            linewidth=2,
        )
    )

    # --------------------------------------------------------
    # Three-point arcs
    # --------------------------------------------------------

    ax.add_patch(
        Arc(
            (
                basket_x_left,
                basket_y,
            ),
            44,
            44,
            angle=0,
            theta1=-68,
            theta2=68,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Arc(
            (
                basket_x_right,
                basket_y,
            ),
            44,
            44,
            angle=0,
            theta1=112,
            theta2=248,
            linewidth=1.5,
        )
    )

    # --------------------------------------------------------
    # Court appearance
    # --------------------------------------------------------

    ax.set_xlim(
        -2,
        COURT_LENGTH + 2,
    )

    ax.set_ylim(
        -2,
        COURT_WIDTH + 2,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Court length (ft)"
    )

    ax.set_ylabel(
        "Court width (ft)"
    )

    ax.grid(
        True,
        alpha=0.15,
    )


# ============================================================
# BUILD VISUALIZATION
# ============================================================

print()
print("Building calibration visualization...")

fig = plt.figure(
    figsize=(16, 9)
)

# ============================================================
# CAMERA FRAME
# ============================================================

ax1 = fig.add_subplot(
    1,
    2,
    1,
)

ax1.imshow(
    cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )
)

ax1.set_title(
    "Broadcast Frame + Detected Court Landmarks"
)

for point, landmark_id in zip(
    image_points_np,
    landmark_ids,
):

    ax1.scatter(
        point[0],
        point[1],
        s=35,
    )

    ax1.text(
        point[0] + 5,
        point[1] - 5,
        str(landmark_id),
        fontsize=8,
    )

ax1.set_xlim(
    0,
    frame_width,
)

ax1.set_ylim(
    frame_height,
    0,
)

ax1.axis(
    "off"
)


# ============================================================
# CALIBRATED COURT
# ============================================================

ax2 = fig.add_subplot(
    1,
    2,
    2,
)

draw_court(ax2)

# ------------------------------------------------------------
# Plot transformed players
# ------------------------------------------------------------

valid_players = players[
    np.isfinite(
        players["debug_court_x"]
    )
    &
    np.isfinite(
        players["debug_court_y"]
    )
]

team_a = valid_players[
    valid_players["team"] == "Team_A"
]

team_b = valid_players[
    valid_players["team"] == "Team_B"
]

excluded = valid_players[
    ~valid_players["team"].isin(
        ["Team_A", "Team_B"]
    )
]

if len(team_a):

    ax2.scatter(
        team_a["debug_court_x"],
        team_a["debug_court_y"],
        s=18,
        alpha=0.35,
        label="Team A",
    )

if len(team_b):

    ax2.scatter(
        team_b["debug_court_x"],
        team_b["debug_court_y"],
        s=18,
        alpha=0.35,
        label="Team B",
    )

if len(excluded):

    ax2.scatter(
        excluded["debug_court_x"],
        excluded["debug_court_y"],
        s=10,
        alpha=0.15,
        label="Excluded",
    )

ax2.set_title(
    "Calibrated Player Positions"
)

ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        "Basketball Court Calibration Debug\n"
        f"Landmarks: {len(image_points_np)} | "
        f"Inliers: {inliers} | "
        f"Mean error: {mean_error:.2f}"
    ),
    fontsize=16,
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("✅ CALIBRATION DEBUG COMPLETE")
print("=" * 70)

print()
print(
    f"Landmarks:       {len(image_points_np)}"
)

print(
    f"Inliers:         {inliers}/{len(image_points_np)}"
)

print(
    f"Mean error:      {mean_error:.2f}"
)

print(
    f"Max error:       {max_error:.2f}"
)

print(
    f"Players plotted: {len(valid_players)}"
)

print()
print(
    f"Output: {OUTPUT_PATH}"
)

print("=" * 70)