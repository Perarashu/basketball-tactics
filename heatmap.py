"""
heatmap.py

Presentation-ready basketball player heatmaps.

Input:
    outputs/player_coordinates_with_teams.csv

Expected columns:
    frame
    player_id
    team
    court_x
    court_y

IMPORTANT:
    court_x and court_y are already calibrated into
    basketball-court coordinates:

        court_x = 0 → 94 feet
        court_y = 0 → 50 feet

    Therefore we DO NOT normalize them again.

Outputs:
    outputs/heatmap_team_a.png
    outputs/heatmap_team_b.png
    outputs/heatmap_combined.png
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle, Arc


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = Path(
    "outputs/player_coordinates_with_teams.csv"
)

OUTPUT_DIR = Path("outputs")

# Real basketball court dimensions
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0

# Heatmap resolution
GRID_X = 188
GRID_Y = 100

# Gaussian smoothing
SIGMA = 7


# ============================================================
# LOAD PLAYER DATA
# ============================================================

def load_data():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "frame",
        "player_id",
        "team",
        "court_x",
        "court_y",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # Convert coordinates safely
    df["court_x"] = pd.to_numeric(
        df["court_x"],
        errors="coerce",
    )

    df["court_y"] = pd.to_numeric(
        df["court_y"],
        errors="coerce",
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "court_x",
            "court_y",
            "team",
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The homography already produces coordinates in feet.
    #
    # Do NOT divide by 640.
    # Do NOT multiply by 94.
    # Do NOT normalize again.
    # --------------------------------------------------------

    df = df[
        (df["court_x"] >= 0)
        & (df["court_x"] <= COURT_LENGTH)
        & (df["court_y"] >= 0)
        & (df["court_y"] <= COURT_WIDTH)
    ]

    return df


# ============================================================
# DRAW BASKETBALL COURT
# ============================================================

def draw_court(ax):
    """
    Draw a clean 94 ft × 50 ft basketball court.

    Coordinate system:

        (0,0)
          ┌──────────────────────────────────┐
          │                                  │
          │                                  │
          │              COURT               │
       50 │                                  │
          │                                  │
          │                                  │
          └──────────────────────────────────┘
          0                                 94
    """

    # --------------------------------------------------------
    # Outer boundary
    # --------------------------------------------------------

    ax.add_patch(
        Rectangle(
            (0, 0),
            COURT_LENGTH,
            COURT_WIDTH,
            fill=False,
            linewidth=2.5,
        )
    )

    # --------------------------------------------------------
    # Half court line
    # --------------------------------------------------------

    ax.plot(
        [COURT_LENGTH / 2, COURT_LENGTH / 2],
        [0, COURT_WIDTH],
        linewidth=2.0,
    )

    # --------------------------------------------------------
    # Center circle
    # --------------------------------------------------------

    ax.add_patch(
        Circle(
            (
                COURT_LENGTH / 2,
                COURT_WIDTH / 2,
            ),
            6,
            fill=False,
            linewidth=2.0,
        )
    )

    # --------------------------------------------------------
    # Center circle small mark
    # --------------------------------------------------------

    ax.scatter(
        [COURT_LENGTH / 2],
        [COURT_WIDTH / 2],
        s=10,
    )

    # --------------------------------------------------------
    # NBA-style key dimensions
    # --------------------------------------------------------

    key_length = 19.0
    key_width = 16.0

    key_y = (
        COURT_WIDTH - key_width
    ) / 2

    # Left key
    ax.add_patch(
        Rectangle(
            (
                0,
                key_y,
            ),
            key_length,
            key_width,
            fill=False,
            linewidth=2.0,
        )
    )

    # Right key
    ax.add_patch(
        Rectangle(
            (
                COURT_LENGTH - key_length,
                key_y,
            ),
            key_length,
            key_width,
            fill=False,
            linewidth=2.0,
        )
    )

    # --------------------------------------------------------
    # Free throw circles
    # --------------------------------------------------------

    ax.add_patch(
        Arc(
            (
                key_length,
                COURT_WIDTH / 2,
            ),
            12,
            12,
            theta1=270,
            theta2=90,
            linewidth=1.8,
        )
    )

    ax.add_patch(
        Arc(
            (
                COURT_LENGTH - key_length,
                COURT_WIDTH / 2,
            ),
            12,
            12,
            theta1=90,
            theta2=270,
            linewidth=1.8,
        )
    )

    # --------------------------------------------------------
    # Restricted areas
    # --------------------------------------------------------

    restricted_radius = 4.0

    ax.add_patch(
        Arc(
            (
                4.0,
                COURT_WIDTH / 2,
            ),
            restricted_radius * 2,
            restricted_radius * 2,
            theta1=270,
            theta2=90,
            linewidth=1.5,
        )
    )

    ax.add_patch(
        Arc(
            (
                COURT_LENGTH - 4.0,
                COURT_WIDTH / 2,
            ),
            restricted_radius * 2,
            restricted_radius * 2,
            theta1=90,
            theta2=270,
            linewidth=1.5,
        )
    )

    # --------------------------------------------------------
    # Baskets
    # --------------------------------------------------------

    ax.scatter(
        [
            4.0,
            COURT_LENGTH - 4.0,
        ],
        [
            COURT_WIDTH / 2,
            COURT_WIDTH / 2,
        ],
        s=35,
        zorder=10,
    )

    # --------------------------------------------------------
    # Three-point arcs
    # --------------------------------------------------------

    three_point_radius = 23.75

    # Left three-point arc
    ax.add_patch(
        Arc(
            (
                4.0,
                COURT_WIDTH / 2,
            ),
            three_point_radius * 2,
            three_point_radius * 2,
            theta1=-67,
            theta2=67,
            linewidth=1.8,
        )
    )

    # Right three-point arc
    ax.add_patch(
        Arc(
            (
                COURT_LENGTH - 4.0,
                COURT_WIDTH / 2,
            ),
            three_point_radius * 2,
            three_point_radius * 2,
            theta1=113,
            theta2=247,
            linewidth=1.8,
        )
    )

    # --------------------------------------------------------
    # Court limits
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        COURT_LENGTH,
    )

    ax.set_ylim(
        0,
        COURT_WIDTH,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.axis("off")


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    df,
    title,
    output_path,
):

    if df.empty:

        print(
            f"⚠️ No data available for {title}"
        )

        return

    x = df["court_x"].to_numpy(
        dtype=float
    )

    y = df["court_y"].to_numpy(
        dtype=float
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Observations: {len(df)}"
    )

    print(
        f"X range: {x.min():.2f} → {x.max():.2f} ft"
    )

    print(
        f"Y range: {y.min():.2f} → {y.max():.2f} ft"
    )

    # --------------------------------------------------------
    # 2D position histogram
    # --------------------------------------------------------

    heatmap, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[
            GRID_X,
            GRID_Y,
        ],
        range=[
            [
                0,
                COURT_LENGTH,
            ],
            [
                0,
                COURT_WIDTH,
            ],
        ],
    )

    # --------------------------------------------------------
    # Smooth the heatmap
    # --------------------------------------------------------

    heatmap = cv2.GaussianBlur(
        heatmap.astype(
            np.float32
        ),
        (
            0,
            0,
        ),
        SIGMA,
    )

    # --------------------------------------------------------
    # Normalize to 0–1
    # --------------------------------------------------------

    maximum = heatmap.max()

    if maximum > 0:

        heatmap = (
            heatmap / maximum
        )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 8),
    )

    # --------------------------------------------------------
    # Heatmap layer
    # --------------------------------------------------------

    ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[
            0,
            COURT_LENGTH,
            0,
            COURT_WIDTH,
        ],
        interpolation="bilinear",
        cmap="hot",
        alpha=0.70,
        zorder=1,
    )

    # --------------------------------------------------------
    # Court layer
    # --------------------------------------------------------

    draw_court(ax)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
        pad=15,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close()

    print(
        f"✅ Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("🏀 BASKETBALL PLAYER HEATMAP")
    print("=" * 60)

    df = load_data()

    print()
    print(
        f"Valid observations: {len(df)}"
    )

    print()
    print("Team distribution:")

    print(
        df["team"].value_counts()
    )

    # --------------------------------------------------------
    # Overall coordinate check
    # --------------------------------------------------------

    print()
    print("Court coordinate ranges:")

    print(
        f"X: "
        f"{df['court_x'].min():.2f}"
        f" → "
        f"{df['court_x'].max():.2f} ft"
    )

    print(
        f"Y: "
        f"{df['court_y'].min():.2f}"
        f" → "
        f"{df['court_y'].max():.2f} ft"
    )

    # --------------------------------------------------------
    # Team A
    # --------------------------------------------------------

    team_a = df[
        df["team"] == "Team_A"
    ]

    create_heatmap(
        team_a,
        "Team A — Player Movement Heatmap",
        OUTPUT_DIR /
        "heatmap_team_a.png",
    )

    # --------------------------------------------------------
    # Team B
    # --------------------------------------------------------

    team_b = df[
        df["team"] == "Team_B"
    ]

    create_heatmap(
        team_b,
        "Team B — Player Movement Heatmap",
        OUTPUT_DIR /
        "heatmap_team_b.png",
    )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    create_heatmap(
        df,
        "Combined — Player Movement Heatmap",
        OUTPUT_DIR /
        "heatmap_combined.png",
    )

    print()
    print("=" * 60)
    print("✅ HEATMAP GENERATION COMPLETE")
    print("=" * 60)

    print()
    print("Generated:")

    print(
        "  outputs/heatmap_team_a.png"
    )

    print(
        "  outputs/heatmap_team_b.png"
    )

    print(
        "  outputs/heatmap_combined.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()