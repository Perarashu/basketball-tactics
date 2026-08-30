"""
team_assignment.py

MVP Team Assignment

Goal:
    Identify the 10 basketball players as:
        Team_A = Green jerseys
        Team_B = White jerseys

Rules:
    - Use player appearance/jersey color.
    - Select exactly 5 green players.
    - Select exactly 5 white players.
    - Ignore tracker IDs that are not part of the 10 players.

Inputs:
    outputs/player_coordinates_v2.csv
    data/videos/game_30fps.mp4

Outputs:
    outputs/player_team_assignments.csv
    outputs/player_coordinates_with_teams.csv
"""

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PLAYER_CSV = "outputs/player_coordinates_v2.csv"
VIDEO_PATH = "data/videos/game_30fps.mp4"

OUTPUT_ASSIGNMENTS = "outputs/player_team_assignments.csv"
OUTPUT_TRACKING = "outputs/player_coordinates_with_teams.csv"

TEAM_A = "Team_A"
TEAM_B = "Team_B"

EXPECTED_PLAYERS_PER_TEAM = 5


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    """
    Find the first matching column from a list of possible names.
    """

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Tried: {candidates}\n"
            f"Available: {df.columns.tolist()}"
        )

    return None


def safe_int(value):
    """
    Convert numpy/pandas numeric values to normal Python int.
    """

    try:
        return int(value)
    except Exception:
        return value


def jersey_features(frame, x, y, width, height):
    """
    Estimate jersey color from the upper-middle body region.

    Returns:
        hue_median
        saturation_median
        value_median
        green_ratio
        white_ratio
    """

    h, w = frame.shape[:2]

    # Clamp center point.
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # --------------------------------------------------------
    # Player box
    # --------------------------------------------------------

    half_w = max(10, int(width * 0.30))
    half_h = max(15, int(height * 0.45))

    x1 = max(0, x - half_w)
    x2 = min(w, x + half_w)

    y1 = max(0, y - half_h)
    y2 = min(h, y + half_h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return 0, 0, 0, 0, 0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # --------------------------------------------------------
    # Focus on central torso region.
    # This reduces shorts/legs/background influence.
    # --------------------------------------------------------

    ch, cw = hsv.shape[:2]

    torso_y1 = int(ch * 0.20)
    torso_y2 = int(ch * 0.70)

    torso_x1 = int(cw * 0.20)
    torso_x2 = int(cw * 0.80)

    torso = hsv[torso_y1:torso_y2, torso_x1:torso_x2]

    if torso.size == 0:
        torso = hsv

    H = torso[:, :, 0]
    S = torso[:, :, 1]
    V = torso[:, :, 2]

    # Flatten.
    H_flat = H.reshape(-1)
    S_flat = S.reshape(-1)
    V_flat = V.reshape(-1)

    # Remove extreme outliers.
    valid = (
        (V_flat > 40)
        & (S_flat >= 0)
    )

    H_flat = H_flat[valid]
    S_flat = S_flat[valid]
    V_flat = V_flat[valid]

    if len(H_flat) == 0:
        return 0, 0, 0, 0, 0

    hue_median = float(np.median(H_flat))
    sat_median = float(np.median(S_flat))
    val_median = float(np.median(V_flat))

    # --------------------------------------------------------
    # OpenCV HSV:
    #
    # Green is approximately hue 35-90.
    #
    # White has:
    #   low saturation
    #   relatively high brightness
    # --------------------------------------------------------

    green_mask = (
        (H_flat >= 35)
        & (H_flat <= 90)
        & (S_flat >= 55)
        & (V_flat >= 45)
    )

    white_mask = (
        (S_flat <= 55)
        & (V_flat >= 140)
    )

    green_ratio = float(np.mean(green_mask))
    white_ratio = float(np.mean(white_mask))

    return (
        hue_median,
        sat_median,
        val_median,
        green_ratio,
        white_ratio,
    )


# ============================================================
# LOAD PLAYER DATA
# ============================================================

print("=" * 60)
print("🏀 TEAM ASSIGNMENT V3")
print("=" * 60)

if not os.path.exists(PLAYER_CSV):
    raise RuntimeError(f"Missing player CSV: {PLAYER_CSV}")

if not os.path.exists(VIDEO_PATH):
    raise RuntimeError(f"Missing video: {VIDEO_PATH}")

players = pd.read_csv(PLAYER_CSV)

print()
print(f"Player rows: {len(players)}")

# ------------------------------------------------------------
# Detect required columns.
# ------------------------------------------------------------

player_id_col = find_column(
    players,
    ["player_id", "tracker_id", "id"]
)

frame_col = find_column(
    players,
    ["frame", "frame_id"]
)

x_col = find_column(
    players,
    [
        "camera_x",
        "center_x",
        "x",
        "cx"
    ]
)

y_col = find_column(
    players,
    [
        "camera_y",
        "center_y",
        "y",
        "cy"
    ]
)

width_col = find_column(
    players,
    [
        "width",
        "bbox_width",
        "w"
    ],
    required=False
)

height_col = find_column(
    players,
    [
        "height",
        "bbox_height",
        "h"
    ],
    required=False
)

# If width/height are unavailable, use reasonable defaults.
if width_col is None:
    players["_width"] = 60
    width_col = "_width"

if height_col is None:
    players["_height"] = 140
    height_col = "_height"


# ============================================================
# PLAYER COUNTS
# ============================================================

counts = (
    players[player_id_col]
    .value_counts()
    .sort_values(ascending=False)
)

print()
print("Player observation counts:")
print(counts)

# ------------------------------------------------------------
# Ignore tiny tracker fragments.
#
# Main players have substantially more observations.
# ------------------------------------------------------------

MIN_OBSERVATIONS = 20

eligible_ids = [
    safe_int(pid)
    for pid, count in counts.items()
    if count >= MIN_OBSERVATIONS
]

print()
print("Eligible IDs:")
print(eligible_ids)

if len(eligible_ids) < 10:
    raise RuntimeError(
        f"Only {len(eligible_ids)} players have enough observations. "
        f"Need at least 10."
    )

# ------------------------------------------------------------
# If there are more than 10 eligible IDs, keep the 10 with
# the most observations.
# ------------------------------------------------------------

if len(eligible_ids) > 10:
    eligible_ids = [
        safe_int(pid)
        for pid in counts.head(10).index
    ]

print()
print("Using 10 main player IDs:")
print(eligible_ids)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print()
print("Video:")
print(f"FPS:       {video_fps:.2f}")
print(f"Frames:    {video_frames}")
print(f"Size:      {video_width} x {video_height}")


# ============================================================
# SAMPLE PLAYER OBSERVATIONS
# ============================================================

print()
print("=" * 60)
print("🎨 EXTRACTING JERSEY FEATURES")
print("=" * 60)

# Store features here.
feature_rows = []

SAMPLES_PER_PLAYER = 30

for player_id in eligible_ids:

    player_data = players[
        players[player_id_col] == player_id
    ].copy()

    if len(player_data) == 0:
        continue

    # --------------------------------------------------------
    # Spread samples through the player's observations.
    # --------------------------------------------------------

    sample_count = min(
        SAMPLES_PER_PLAYER,
        len(player_data)
    )

    sample_indices = np.linspace(
        0,
        len(player_data) - 1,
        sample_count
    ).astype(int)

    samples = player_data.iloc[sample_indices]

    player_features = []

    for _, row in samples.iterrows():

        frame_number = safe_int(row[frame_col])

        x = row[x_col]
        y = row[y_col]

        width = row[width_col]
        height = row[height_col]

        if pd.isna(x) or pd.isna(y):
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        features = jersey_features(
            frame,
            x,
            y,
            width,
            height
        )

        player_features.append(features)

    if not player_features:
        print(
            f"Player {player_id:2d} | "
            "No valid samples"
        )
        continue

    feature_array = np.array(
        player_features,
        dtype=float
    )

    hue = float(
        np.median(feature_array[:, 0])
    )

    saturation = float(
        np.median(feature_array[:, 1])
    )

    value = float(
        np.median(feature_array[:, 2])
    )

    green_ratio = float(
        np.median(feature_array[:, 3])
    )

    white_ratio = float(
        np.median(feature_array[:, 4])
    )

    feature_rows.append(
        {
            "player_id": player_id,
            "observations": int(
                len(player_data)
            ),
            "hue": hue,
            "saturation": saturation,
            "value": value,
            "green_ratio": green_ratio,
            "white_ratio": white_ratio,
        }
    )

    print(
        f"Player {player_id:2d} | "
        f"samples={len(player_features):2d} | "
        f"H={hue:5.1f} | "
        f"S={saturation:5.1f} | "
        f"V={value:5.1f} | "
        f"green={green_ratio:.2f} | "
        f"white={white_ratio:.2f}"
    )

cap.release()

features_df = pd.DataFrame(feature_rows)

if len(features_df) < 10:
    raise RuntimeError(
        "Could not extract jersey features for 10 players."
    )


# ============================================================
# CLASSIFY GREEN VS WHITE
# ============================================================

print()
print("=" * 60)
print("🟢⚪ CLASSIFYING JERSEY COLORS")
print("=" * 60)

# ------------------------------------------------------------
# Green score.
#
# Strong green_ratio is the main signal.
# Hue around 60-80 is also strong evidence.
# ------------------------------------------------------------

def green_score(row):

    green_ratio = row["green_ratio"]
    hue = row["hue"]

    # Hue score:
    # 60-75 = very green.
    hue_distance = abs(hue - 67.5)

    hue_score = max(
        0.0,
        1.0 - hue_distance / 35.0
    )

    # White jerseys should not score green merely because
    # of small amounts of green background.
    score = (
        0.75 * green_ratio
        + 0.25 * hue_score
    )

    # If saturation is extremely low, suppress green score.
    if row["saturation"] < 50:
        score *= 0.25

    return float(score)


features_df["green_score"] = features_df.apply(
    green_score,
    axis=1
)

features_df = features_df.sort_values(
    "green_score",
    ascending=False
).reset_index(drop=True)

print()
print(
    features_df[
        [
            "player_id",
            "observations",
            "hue",
            "saturation",
            "green_ratio",
            "white_ratio",
            "green_score",
        ]
    ].to_string(index=False)
)


# ============================================================
# FORCE EXACTLY 5 VS 5
# ============================================================

print()
print("=" * 60)
print("👥 SELECTING EXACTLY 5 VS 5")
print("=" * 60)

green_players = (
    features_df
    .head(EXPECTED_PLAYERS_PER_TEAM)
    ["player_id"]
    .tolist()
)

white_players = (
    features_df
    .iloc[EXPECTED_PLAYERS_PER_TEAM:
          EXPECTED_PLAYERS_PER_TEAM * 2]
    ["player_id"]
    .tolist()
)

green_players = [
    safe_int(x)
    for x in green_players
]

white_players = [
    safe_int(x)
    for x in white_players
]

print()
print("🟢 Green Team A:")
print(green_players)

print()
print("⚪ White Team B:")
print(white_players)


# ============================================================
# VALIDATION
# ============================================================

if len(green_players) != 5:
    raise RuntimeError(
        f"Green team has {len(green_players)} players."
    )

if len(white_players) != 5:
    raise RuntimeError(
        f"White team has {len(white_players)} players."
    )

if set(green_players) & set(white_players):
    raise RuntimeError(
        "Player appears in both teams."
    )


# ============================================================
# ASSIGNMENT TABLE
# ============================================================

team_map = {}

for player_id in green_players:
    team_map[player_id] = TEAM_A

for player_id in white_players:
    team_map[player_id] = TEAM_B


assignment_rows = []

for _, row in features_df.iterrows():

    player_id = safe_int(
        row["player_id"]
    )

    if player_id in green_players:
        team = TEAM_A
        jersey_color = "GREEN"

    elif player_id in white_players:
        team = TEAM_B
        jersey_color = "WHITE"

    else:
        team = "EXCLUDED"
        jersey_color = "UNKNOWN"

    assignment_rows.append(
        {
            "player_id": player_id,
            "team": team,
            "jersey_color": jersey_color,
            "observations": row["observations"],
            "hue": row["hue"],
            "saturation": row["saturation"],
            "value": row["value"],
            "green_ratio": row["green_ratio"],
            "white_ratio": row["white_ratio"],
            "green_score": row["green_score"],
        }
    )

assignments_df = pd.DataFrame(
    assignment_rows
)

# Sort by player ID for easier downstream use.
assignments_df = assignments_df.sort_values(
    "player_id"
).reset_index(drop=True)


# ============================================================
# SAVE ASSIGNMENTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_ASSIGNMENTS),
    exist_ok=True
)

assignments_df.to_csv(
    OUTPUT_ASSIGNMENTS,
    index=False
)


# ============================================================
# ADD TEAM TO EVERY TRACKING ROW
# ============================================================

print()
print("Adding team labels to tracking data...")

tracking = players.copy()

tracking["player_id"] = tracking[
    player_id_col
].apply(safe_int)

tracking["team"] = (
    tracking["player_id"]
    .map(team_map)
    .fillna("EXCLUDED")
)

tracking["jersey_color"] = (
    tracking["team"]
    .map(
        {
            TEAM_A: "GREEN",
            TEAM_B: "WHITE",
        }
    )
    .fillna("UNKNOWN")
)

os.makedirs(
    os.path.dirname(OUTPUT_TRACKING),
    exist_ok=True
)

tracking.to_csv(
    OUTPUT_TRACKING,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("📊 FINAL TEAM ASSIGNMENT")
print("=" * 60)

display_columns = [
    "player_id",
    "team",
    "jersey_color",
    "observations",
    "hue",
    "green_score",
]

print(
    assignments_df[
        display_columns
    ].to_string(index=False)
)

print()
print("=" * 60)
print("TEAM COUNTS")
print("=" * 60)

print(
    assignments_df["team"].value_counts()
)

print()
print("=" * 60)
print("TEAM MEMBERS")
print("=" * 60)

print()
print(
    f"Green Team A: {green_players}"
)

print(
    f"White Team B: {white_players}"
)

print()
print("=" * 60)
print("✅ TEAM ASSIGNMENT COMPLETE")
print("=" * 60)

print(
    f"Green Team A: {len(green_players)}"
)

print(
    f"White Team B: {len(white_players)}"
)

print()
print(
    f"Assignments: {OUTPUT_ASSIGNMENTS}"
)

print(
    f"Tracking:    {OUTPUT_TRACKING}"
)

print("=" * 60)