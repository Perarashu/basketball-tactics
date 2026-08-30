import pandas as pd
import matplotlib.pyplot as plt

# Load coordinates
df = pd.read_csv("outputs/player_coordinates.csv")

print("📊 Loaded coordinate data")
print(f"Observations: {len(df)}")
print(f"Players: {df['player_id'].nunique()}")

# Create plot
plt.figure(figsize=(12, 7))

# Plot each tracked player
for player_id in df["player_id"].unique():

    player = df[
        df["player_id"] == player_id
    ]

    plt.plot(
        player["x"],
        player["y"],
        marker="o",
        markersize=2,
        linewidth=1,
        label=f"Player {player_id}"
    )

# Broadcast coordinate system
plt.xlim(0, 1920)
plt.ylim(1080, 0)

plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")

plt.title(
    "Player Movement — Broadcast Coordinates"
)

plt.grid(alpha=0.2)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "outputs/player_movement.png",
    dpi=150
)

plt.show()

print(
    "✅ Saved: outputs/player_movement.png"
)