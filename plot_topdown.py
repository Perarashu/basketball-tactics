import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "outputs/player_coordinates.csv"
OUTPUT_PATH = "outputs/topdown_coordinates.png"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("===================================")
print("🏀 TOP-DOWN COORDINATE VISUALIZER")
print("===================================")
print(f"Rows: {len(df)}")
print(f"Players: {df['player_id'].nunique()}")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Plot each player
for player_id, player_df in df.groupby("player_id"):

    ax.scatter(
        player_df["court_x"],
        player_df["court_y"],
        s=12,
        alpha=0.45,
        label=f"Player {player_id}"
    )

# ============================================================
# NORMALIZED COURT
# ============================================================

# Court boundary
ax.plot(
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    linewidth=2
)

# Half court
ax.plot(
    [0.5, 0.5],
    [0, 1],
    linewidth=1
)

# ============================================================
# COURT MARKERS
# ============================================================

# Left/right baskets
ax.scatter(
    [0, 1],
    [0.5, 0.5],
    s=80,
    marker="o"
)

# ============================================================
# FORMAT
# ============================================================

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.15, 1.35)

ax.set_aspect("equal")

ax.set_xlabel("Court X")
ax.set_ylabel("Court Y")

ax.set_title(
    "Player Positions — Homography Test"
)

ax.grid(True, alpha=0.25)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=8
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print()
print("===================================")
print("✅ TOP-DOWN PLOT COMPLETE")
print("===================================")
print(f"Saved: {OUTPUT_PATH}")
print("===================================")