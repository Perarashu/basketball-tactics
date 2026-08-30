import cv2
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

VIDEO = "data/videos/game_30fps.mp4"
POSSESSION_CSV = "outputs/possession_v4.csv"
OUTPUT = "outputs/possession_visualization.mp4"

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("🏀 POSSESSION VISUALIZATION")
print("=" * 60)

possession = pd.read_csv(POSSESSION_CSV)

print(f"Possession rows: {len(possession)}")
print("Columns:", list(possession.columns))

# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps:.2f} FPS")
print(f"Frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# POSSESSION LOOKUP
# ============================================================

possession_lookup = {}

for _, row in possession.iterrows():

    frame = int(row["frame"])

    player = row.get("possession_player")

    if pd.isna(player):
        player = None
    else:
        player = int(player)

    possession_lookup[frame] = player

# ============================================================
# DRAW
# ============================================================

frame_number = 0

while True:

    ok, frame = cap.read()

    if not ok:
        break

    player = possession_lookup.get(frame_number)

    # --------------------------------------------------------
    # TOP STATUS
    # --------------------------------------------------------

    if player is not None:

        text = f"BALL POSSESSION: PLAYER {player}"

    else:

        text = "BALL POSSESSION: UNKNOWN"

    cv2.rectangle(
        frame,
        (20, 20),
        (600, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(frame)

    frame_number += 1

# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()

print()
print("=" * 60)
print("✅ POSSESSION VISUALIZATION COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT}")
print("=" * 60)