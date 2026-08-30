import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PLAYER_CSV = "outputs/player_coordinates_with_teams.csv"
TEAM_CSV = "outputs/player_team_assignments.csv"
BALL_CSV = "outputs/ball_tracking_v2.csv"
POSSESSION_CSV = "outputs/possession_v6.csv"

OUTPUT_CSV = "outputs/tactical_events.csv"

TEAM_A = "Team_A"
TEAM_B = "Team_B"

EXCLUDED_IDS = {8}

VALID_TEAMS = {TEAM_A, TEAM_B}


# ============================================================
# HELPERS
# ============================================================

def get_column(df, names, required=True):

    for name in names:

        if name in df.columns:
            return name

    if required:

        raise RuntimeError(
            f"Missing required column.\n"
            f"Expected one of: {names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("🏀 BUILDING TEAM-AWARE TACTICAL EVENTS")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    PLAYER_CSV,
    TEAM_CSV,
    BALL_CSV,
    POSSESSION_CSV
]:

    if not os.path.exists(path):

        raise RuntimeError(
            f"Missing required file: {path}"
        )


# ============================================================
# LOAD
# ============================================================

players = pd.read_csv(
    PLAYER_CSV
)

teams = pd.read_csv(
    TEAM_CSV
)

ball = pd.read_csv(
    BALL_CSV
)

possession = pd.read_csv(
    POSSESSION_CSV
)


print()
print("Loaded:")
print(f"Players:     {len(players)}")
print(f"Teams:       {len(teams)}")
print(f"Ball:        {len(ball)}")
print(f"Possession:  {len(possession)}")


# ============================================================
# PLAYER DATA
# ============================================================

player_id_col = get_column(
    players,
    ["player_id", "object_id"]
)

players = players.rename(
    columns={
        player_id_col: "player_id"
    }
)

players["player_id"] = pd.to_numeric(
    players["player_id"],
    errors="coerce"
)

players["frame"] = pd.to_numeric(
    players["frame"],
    errors="coerce"
)

players["court_x"] = pd.to_numeric(
    players["court_x"],
    errors="coerce"
)

players["court_y"] = pd.to_numeric(
    players["court_y"],
    errors="coerce"
)

players = players.dropna(
    subset=[
        "player_id",
        "frame"
    ]
)

players["player_id"] = (
    players["player_id"]
    .astype(int)
)

players["frame"] = (
    players["frame"]
    .astype(int)
)


# ============================================================
# TEAM ASSIGNMENTS
# ============================================================

team_player_col = get_column(
    teams,
    ["player_id", "object_id"]
)

team_col = get_column(
    teams,
    ["team"]
)

teams = teams.rename(
    columns={
        team_player_col: "player_id",
        team_col: "assigned_team"
    }
)

teams["player_id"] = pd.to_numeric(
    teams["player_id"],
    errors="coerce"
)

teams = teams.dropna(
    subset=["player_id"]
)

teams["player_id"] = (
    teams["player_id"]
    .astype(int)
)

teams["assigned_team"] = (
    teams["assigned_team"]
    .astype(str)
    .str.strip()
)


# Only real teams.
teams = teams[
    teams["assigned_team"].isin(
        VALID_TEAMS
    )
].copy()


# Remove excluded tracker.
teams = teams[
    ~teams["player_id"].isin(
        EXCLUDED_IDS
    )
].copy()


# One assignment per player.
teams = (
    teams[
        [
            "player_id",
            "assigned_team"
        ]
    ]
    .drop_duplicates(
        subset=["player_id"]
    )
)


# ============================================================
# VERIFY 5 VS 5
# ============================================================

print()
print("Team assignments:")

team_a_ids = sorted(
    teams.loc[
        teams["assigned_team"] == TEAM_A,
        "player_id"
    ].unique()
)

team_b_ids = sorted(
    teams.loc[
        teams["assigned_team"] == TEAM_B,
        "player_id"
    ].unique()
)

print(
    f"Team_A: {team_a_ids}"
)

print(
    f"Team_B: {team_b_ids}"
)


if len(team_a_ids) != 5:

    raise RuntimeError(
        f"Team_A does not contain 5 players: "
        f"{team_a_ids}"
    )


if len(team_b_ids) != 5:

    raise RuntimeError(
        f"Team_B does not contain 5 players: "
        f"{team_b_ids}"
    )


# ============================================================
# IMPORTANT:
# PLAYER CSV MAY ALREADY HAVE TEAM COLUMN
# ============================================================

# Delete any existing team-related columns before merging.
for column in [
    "team",
    "assigned_team",
    "team_x",
    "team_y"
]:

    if column in players.columns:

        players = players.drop(
            columns=[column]
        )


# Merge exactly one team column.
players = players.merge(
    teams,
    on="player_id",
    how="inner"
)

players = players.rename(
    columns={
        "assigned_team": "team"
    }
)


# ============================================================
# REMOVE EXCLUDED PLAYERS
# ============================================================

players = players[
    ~players["player_id"].isin(
        EXCLUDED_IDS
    )
].copy()


# ============================================================
# BALL DATA
# ============================================================

ball_frame_col = get_column(
    ball,
    ["frame"]
)

ball_x_col = get_column(
    ball,
    ["court_x"]
)

ball_y_col = get_column(
    ball,
    ["court_y"]
)

ball = ball.rename(
    columns={
        ball_frame_col: "frame",
        ball_x_col: "ball_x",
        ball_y_col: "ball_y"
    }
)

ball["frame"] = pd.to_numeric(
    ball["frame"],
    errors="coerce"
)

ball["ball_x"] = pd.to_numeric(
    ball["ball_x"],
    errors="coerce"
)

ball["ball_y"] = pd.to_numeric(
    ball["ball_y"],
    errors="coerce"
)

ball = ball.dropna(
    subset=[
        "frame",
        "ball_x",
        "ball_y"
    ]
)

ball["frame"] = (
    ball["frame"]
    .astype(int)
)

ball = (
    ball
    .sort_values("frame")
    .drop_duplicates(
        subset=["frame"],
        keep="last"
    )
)


# ============================================================
# POSSESSION DATA
# ============================================================

pos_frame_col = get_column(
    possession,
    ["frame"]
)

pos_player_col = get_column(
    possession,
    [
        "possession_player",
        "nearest_player"
    ]
)

possession = possession.rename(
    columns={
        pos_frame_col: "frame",
        pos_player_col: "possession_player"
    }
)

possession["frame"] = pd.to_numeric(
    possession["frame"],
    errors="coerce"
)

possession["possession_player"] = pd.to_numeric(
    possession["possession_player"],
    errors="coerce"
)

possession["frame"] = (
    possession["frame"]
    .astype("Int64")
)

# Player 8 is excluded.
possession.loc[
    possession["possession_player"].isin(
        EXCLUDED_IDS
    ),
    "possession_player"
] = np.nan


# ============================================================
# ADD TEAM TO POSSESSION
# ============================================================

team_lookup = (
    teams
    .set_index("player_id")[
        "assigned_team"
    ]
    .to_dict()
)

possession["possession_team"] = (
    possession["possession_player"]
    .map(team_lookup)
)


# Only valid teams.
invalid_team_mask = (
    possession["possession_team"].notna()
    &
    ~possession["possession_team"].isin(
        VALID_TEAMS
    )
)

possession.loc[
    invalid_team_mask,
    [
        "possession_player",
        "possession_team"
    ]
] = np.nan


# ============================================================
# ONE POSSESSION ROW PER FRAME
# ============================================================

possession = (
    possession
    .sort_values("frame")
    .drop_duplicates(
        subset=["frame"],
        keep="last"
    )
)


# ============================================================
# FRAME RANGE
# ============================================================

min_frame = int(
    min(
        players["frame"].min(),
        ball["frame"].min(),
        possession["frame"].min()
    )
)

max_frame = int(
    max(
        players["frame"].max(),
        ball["frame"].max(),
        possession["frame"].max()
    )
)


print()
print("Building frame table...")


frames = pd.DataFrame({
    "frame": np.arange(
        min_frame,
        max_frame + 1
    )
})


# ============================================================
# ADD BALL
# ============================================================

frames = frames.merge(
    ball[
        [
            "frame",
            "ball_x",
            "ball_y"
        ]
    ],
    on="frame",
    how="left"
)

frames["ball_available"] = (
    frames["ball_x"].notna()
    &
    frames["ball_y"].notna()
)


# ============================================================
# ADD POSSESSION
# ============================================================

frames = frames.merge(
    possession[
        [
            "frame",
            "possession_player",
            "possession_team"
        ]
    ],
    on="frame",
    how="left"
)


# ============================================================
# FORCE INTEGER PLAYER IDS
# ============================================================

frames["possession_player"] = pd.to_numeric(
    frames["possession_player"],
    errors="coerce"
)


# ============================================================
# EVENT TYPE
# ============================================================

frames["event_type"] = np.where(
    frames["possession_player"].notna(),
    "POSSESSION",
    "NO_POSSESSION"
)


# ============================================================
# ADD POSSESSOR POSITION
# ============================================================

possessor_positions = players[
    [
        "frame",
        "player_id",
        "court_x",
        "court_y",
        "team"
    ]
].copy()

possessor_positions = possessor_positions.rename(
    columns={
        "player_id": "possession_player",
        "court_x": "possessor_x",
        "court_y": "possessor_y",
        "team": "possessor_team"
    }
)


frames = frames.merge(
    possessor_positions,
    on=[
        "frame",
        "possession_player"
    ],
    how="left"
)


# ============================================================
# POSSESSION FLAGS
# ============================================================

frames["team_a_possession"] = (
    frames["possession_team"]
    == TEAM_A
)

frames["team_b_possession"] = (
    frames["possession_team"]
    == TEAM_B
)


# ============================================================
# PREVIOUS POSSESSION
# ============================================================

frames["previous_possession_player"] = (
    frames["possession_player"]
    .shift(1)
)

frames["previous_possession_team"] = (
    frames["possession_team"]
    .shift(1)
)


# ============================================================
# POSSESSION CHANGE
# ============================================================

frames["possession_change"] = (
    frames["possession_player"].notna()
    &
    frames["previous_possession_player"].notna()
    &
    (
        frames["possession_player"]
        !=
        frames["previous_possession_player"]
    )
)


# ============================================================
# TEAM CHANGE
# ============================================================

frames["team_change"] = (
    frames["possession_team"].notna()
    &
    frames["previous_possession_team"].notna()
    &
    (
        frames["possession_team"]
        !=
        frames["previous_possession_team"]
    )
)


# ============================================================
# COURT ZONE
# ============================================================

def court_zone(x, y):

    if pd.isna(x) or pd.isna(y):

        return "UNKNOWN"

    x = float(x)
    y = float(y)

    # Broad zones only.
    # We will calibrate these after visual inspection.

    if x < 150:
        horizontal = "LEFT"

    elif x > 450:
        horizontal = "RIGHT"

    else:
        horizontal = "CENTER"


    if y < 220:
        vertical = "TOP"

    elif y > 450:
        vertical = "BOTTOM"

    else:
        vertical = "MID"


    return (
        f"{vertical}_{horizontal}"
    )


frames["ball_zone"] = [
    court_zone(x, y)
    for x, y in zip(
        frames["ball_x"],
        frames["ball_y"]
    )
]


frames["possessor_zone"] = [
    court_zone(x, y)
    for x, y in zip(
        frames["possessor_x"],
        frames["possessor_y"]
    )
]


# ============================================================
# PASS DATA
# ============================================================

frames["pass_event"] = False
frames["pass_from_player"] = np.nan
frames["pass_to_player"] = np.nan
frames["pass_from_team"] = np.nan
frames["pass_to_team"] = np.nan
frames["pass_confidence"] = np.nan


# We currently have no reliable pass detector.
# Therefore we do NOT manufacture passes here.


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

frames = frames[
    [
        "frame",

        "ball_x",
        "ball_y",
        "ball_available",
        "ball_zone",

        "possession_player",
        "possession_team",

        "possessor_x",
        "possessor_y",
        "possessor_team",
        "possessor_zone",

        "event_type",

        "previous_possession_player",
        "previous_possession_team",

        "possession_change",
        "team_change",

        "team_a_possession",
        "team_b_possession",

        "pass_event",
        "pass_from_player",
        "pass_to_player",
        "pass_from_team",
        "pass_to_team",
        "pass_confidence"
    ]
]


# ============================================================
# SAVE
# ============================================================

frames.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("📊 TEAM-AWARE TACTICAL EVENTS")
print("=" * 60)

print(
    f"Frames:              {len(frames)}"
)

print(
    f"Ball available:      "
    f"{frames['ball_available'].sum()}"
)

print(
    f"Possession frames:   "
    f"{frames['possession_player'].notna().sum()}"
)

print(
    f"Team A possession:   "
    f"{frames['team_a_possession'].sum()}"
)

print(
    f"Team B possession:   "
    f"{frames['team_b_possession'].sum()}"
)

print(
    f"Possession changes:  "
    f"{frames['possession_change'].sum()}"
)

print(
    f"Team changes:        "
    f"{frames['team_change'].sum()}"
)


# ============================================================
# POSSESSION BY TEAM
# ============================================================

print()
print("Possession by team:")

team_counts = (
    frames[
        frames["possession_team"].notna()
    ]
    .groupby(
        "possession_team"
    )
    .size()
)

print(
    team_counts
)


# ============================================================
# POSSESSION BY PLAYER
# ============================================================

print()
print("Possession by player:")

player_counts = (
    frames[
        frames["possession_player"].notna()
    ]
    .groupby(
        "possession_player"
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    player_counts
)


# ============================================================
# TEAM CHANGES
# ============================================================

print()
print("Team possession changes:")

changes = frames[
    frames["team_change"]
].copy()


if len(changes):

    for _, row in changes.iterrows():

        old_team = (
            row["previous_possession_team"]
        )

        new_team = (
            row["possession_team"]
        )

        old_player = (
            row["previous_possession_player"]
        )

        new_player = (
            row["possession_player"]
        )

        print(
            f"Frame {int(row['frame']):3d}: "
            f"{old_team} "
            f"({int(old_player)}) "
            f"-> "
            f"{new_team} "
            f"({int(new_player)})"
        )

else:

    print(
        "No team changes detected."
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("✅ TEAM-AWARE TACTICAL EVENTS COMPLETE")
print("=" * 60)

print(
    f"Output: {OUTPUT_CSV}"
)

print("=" * 60)