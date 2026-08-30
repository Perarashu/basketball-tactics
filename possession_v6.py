import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIG
# ============================================================

PLAYER_CSV = "outputs/player_coordinates_with_teams.csv"
BALL_CSV = "outputs/ball_tracking_v2.csv"

OUTPUT_CSV = "outputs/possession_v6.csv"

MAX_POSSESSION_DISTANCE = 75.0

# Minimum distance advantage required before switching possession.
# This prevents noisy player-ID changes.
MIN_DISTANCE_ADVANTAGE = 8.0

# Number of consecutive observations needed to confirm a
# new possessor.
MIN_CONFIRM_FRAMES = 2

# Ignore extremely small ball movements.
MIN_BALL_MOVEMENT = 2.0


# ============================================================
# TEAM DEFINITIONS
# ============================================================

TEAM_A = "Team_A"   # GREEN
TEAM_B = "Team_B"   # WHITE


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("🏀 POSSESSION V6")
print("=" * 60)


# ============================================================
# LOAD FILES
# ============================================================

if not os.path.exists(PLAYER_CSV):
    raise RuntimeError(f"Missing: {PLAYER_CSV}")

if not os.path.exists(BALL_CSV):
    raise RuntimeError(f"Missing: {BALL_CSV}")


players = pd.read_csv(PLAYER_CSV)
ball = pd.read_csv(BALL_CSV)


print()
print("Loaded:")
print(f"Players:       {len(players)}")
print(f"Ball rows:     {len(ball)}")


# ============================================================
# CHECK PLAYER COLUMNS
# ============================================================

required_player_columns = [
    "frame",
    "player_id",
    "court_x",
    "court_y",
    "team"
]

for column in required_player_columns:

    if column not in players.columns:
        raise RuntimeError(
            f"player_coordinates_with_teams.csv is missing column: {column}"
        )


required_ball_columns = [
    "frame",
    "court_x",
    "court_y"
]

for column in required_ball_columns:

    if column not in ball.columns:
        raise RuntimeError(
            f"ball_tracking_v2.csv is missing column: {column}"
        )


# ============================================================
# NORMALIZE DATA
# ============================================================

players["frame"] = pd.to_numeric(
    players["frame"],
    errors="coerce"
)

players["player_id"] = pd.to_numeric(
    players["player_id"],
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
        "frame",
        "player_id",
        "court_x",
        "court_y"
    ]
)

players["frame"] = players["frame"].astype(int)
players["player_id"] = players["player_id"].astype(int)


ball["frame"] = pd.to_numeric(
    ball["frame"],
    errors="coerce"
)

ball["court_x"] = pd.to_numeric(
    ball["court_x"],
    errors="coerce"
)

ball["court_y"] = pd.to_numeric(
    ball["court_y"],
    errors="coerce"
)

ball = ball.dropna(
    subset=[
        "frame",
        "court_x",
        "court_y"
    ]
)

ball["frame"] = ball["frame"].astype(int)

ball = ball.sort_values(
    "frame"
).reset_index(drop=True)


# ============================================================
# REMOVE EXCLUDED PLAYERS
# ============================================================

players = players[
    players["team"].isin(
        [TEAM_A, TEAM_B]
    )
].copy()


# ============================================================
# TEAM SUMMARY
# ============================================================

print()
print("Team assignments:")

for team in [TEAM_A, TEAM_B]:

    ids = sorted(
        players.loc[
            players["team"] == team,
            "player_id"
        ].unique()
    )

    print(f"{team}: {ids}")


# ============================================================
# PLAYER LOOKUP
# ============================================================

player_lookup = {}

for frame, group in players.groupby("frame"):

    frame_players = []

    for _, row in group.iterrows():

        frame_players.append({
            "player_id": int(row["player_id"]),
            "team": row["team"],
            "x": float(row["court_x"]),
            "y": float(row["court_y"])
        })

    player_lookup[int(frame)] = frame_players


# ============================================================
# BALL MOVEMENT
# ============================================================

ball["prev_x"] = ball["court_x"].shift(1)
ball["prev_y"] = ball["court_y"].shift(1)

ball["ball_movement"] = np.sqrt(
    (
        ball["court_x"] -
        ball["prev_x"]
    ) ** 2
    +
    (
        ball["court_y"] -
        ball["prev_y"]
    ) ** 2
)


# ============================================================
# FIND NEAREST PLAYERS
# ============================================================

print()
print("Finding nearest players...")


nearest_players = []
nearest_distances = []
second_players = []
second_distances = []


for _, row in ball.iterrows():

    frame = int(row["frame"])
    bx = float(row["court_x"])
    by = float(row["court_y"])

    candidates = player_lookup.get(
        frame,
        []
    )

    distances = []

    for player in candidates:

        distance = np.sqrt(
            (player["x"] - bx) ** 2
            +
            (player["y"] - by) ** 2
        )

        distances.append({
            "player_id": player["player_id"],
            "team": player["team"],
            "distance": distance
        })


    distances.sort(
        key=lambda x: x["distance"]
    )


    if len(distances) == 0:

        nearest_players.append(np.nan)
        nearest_distances.append(np.nan)

        second_players.append(np.nan)
        second_distances.append(np.nan)

    else:

        nearest_players.append(
            distances[0]["player_id"]
        )

        nearest_distances.append(
            distances[0]["distance"]
        )

        if len(distances) > 1:

            second_players.append(
                distances[1]["player_id"]
            )

            second_distances.append(
                distances[1]["distance"]
            )

        else:

            second_players.append(np.nan)
            second_distances.append(np.nan)


ball["nearest_player"] = nearest_players
ball["nearest_distance"] = nearest_distances

ball["second_player"] = second_players
ball["second_distance"] = second_distances


# ============================================================
# INITIAL POSSESSION ESTIMATION
# ============================================================

print("Estimating possession...")


raw_possession = []

for _, row in ball.iterrows():

    nearest = row["nearest_player"]
    distance = row["nearest_distance"]

    if pd.isna(nearest):
        raw_possession.append(np.nan)
        continue

    if pd.isna(distance):
        raw_possession.append(np.nan)
        continue

    if distance > MAX_POSSESSION_DISTANCE:
        raw_possession.append(np.nan)
        continue

    # If another player is almost equally close,
    # possession is uncertain.
    second_distance = row["second_distance"]

    if not pd.isna(second_distance):

        advantage = (
            second_distance -
            distance
        )

        if advantage < MIN_DISTANCE_ADVANTAGE:

            raw_possession.append(np.nan)
            continue


    raw_possession.append(
        int(nearest)
    )


ball["raw_possession"] = raw_possession


# ============================================================
# TEMPORAL POSSESSION FILTER
# ============================================================

print("Applying temporal smoothing...")


final_possession = []

current_player = None

candidate_player = None
candidate_count = 0


for i in range(len(ball)):

    proposed = ball.iloc[i]["raw_possession"]

    if pd.isna(proposed):

        # Keep current possession through a short
        # uncertain interval.
        final_possession.append(
            current_player
        )

        continue


    proposed = int(proposed)


    # No current possession yet.
    if current_player is None:

        current_player = proposed

        candidate_player = None
        candidate_count = 0

        final_possession.append(
            current_player
        )

        continue


    # Same player.
    if proposed == current_player:

        candidate_player = None
        candidate_count = 0

        final_possession.append(
            current_player
        )

        continue


    # New candidate player.
    if proposed != candidate_player:

        candidate_player = proposed
        candidate_count = 1

    else:

        candidate_count += 1


    # Confirm new possession only after repeated evidence.
    if candidate_count >= MIN_CONFIRM_FRAMES:

        current_player = candidate_player

        candidate_player = None
        candidate_count = 0


    final_possession.append(
        current_player
    )


ball["possession_player"] = final_possession


# ============================================================
# ADD POSSESSION TEAM
# ============================================================

player_team_map = (
    players[
        ["player_id", "team"]
    ]
    .drop_duplicates(
        "player_id"
    )
    .set_index("player_id")["team"]
    .to_dict()
)


ball["possession_team"] = (
    ball["possession_player"]
    .map(player_team_map)
)


# ============================================================
# REMOVE POSSESSION DURING LARGE BALL GAPS
# ============================================================

for i in range(len(ball)):

    movement = ball.iloc[i]["ball_movement"]

    if pd.isna(movement):
        continue

    # If ball suddenly jumps a very large amount,
    # do not invent a possession change from that single jump.
    if movement > 150:

        if i > 0:

            previous_possession = (
                ball.iloc[i - 1]["possession_player"]
            )

            ball.loc[
                ball.index[i],
                "possession_player"
            ] = previous_possession

            ball.loc[
                ball.index[i],
                "possession_team"
            ] = player_team_map.get(
                previous_possession,
                np.nan
            )


# ============================================================
# BUILD POSSESSION SEGMENTS
# ============================================================

print("Building possession segments...")


segments = []

active_player = None
active_team = None
start_frame = None


for _, row in ball.iterrows():

    frame = int(row["frame"])

    player = row["possession_player"]

    if pd.isna(player):

        player = None

    else:

        player = int(player)


    team = row["possession_team"]

    if pd.isna(team):

        team = None


    # Start first segment.
    if active_player is None:

        if player is not None:

            active_player = player
            active_team = team
            start_frame = frame

        continue


    # Same player.
    if player == active_player:

        continue


    # Player changed.
    if player is not None:

        segments.append({
            "player_id": active_player,
            "team": active_team,
            "start_frame": start_frame,
            "end_frame": frame - 1
        })

        active_player = player
        active_team = team
        start_frame = frame


# Close final segment.
if active_player is not None:

    segments.append({
        "player_id": active_player,
        "team": active_team,
        "start_frame": start_frame,
        "end_frame": int(
            ball.iloc[-1]["frame"]
        )
    })


segments_df = pd.DataFrame(
    segments
)


# ============================================================
# REMOVE VERY SHORT SEGMENTS
# ============================================================

clean_segments = []

for _, segment in segments_df.iterrows():

    length = (
        segment["end_frame"]
        -
        segment["start_frame"]
        +
        1
    )

    # Keep all meaningful segments.
    # One-frame segments are allowed only if they are
    # followed by a genuine team/player change.
    if length >= 2:

        clean_segments.append(
            segment.to_dict()
        )


segments_df = pd.DataFrame(
    clean_segments
)


# ============================================================
# REBUILD POSSESSION FROM CLEAN SEGMENTS
# ============================================================
ball["possession_player"] = pd.Series(
    np.nan,
    index=ball.index,
    dtype="float64"
)

ball["possession_team"] = pd.Series(
    np.nan,
    index=ball.index,
    dtype="object"
)


for _, segment in segments_df.iterrows():

    mask = (
        (ball["frame"] >= segment["start_frame"])
        &
        (ball["frame"] <= segment["end_frame"])
    )

    ball.loc[
        mask,
        "possession_player"
    ] = int(segment["player_id"])

    ball.loc[
        mask,
        "possession_team"
    ] = segment["team"]


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "frame",
    "court_x",
    "court_y",
    "ball_movement",
    "nearest_player",
    "nearest_distance",
    "second_player",
    "second_distance",
    "possession_player",
    "possession_team"
]


ball[output_columns].to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("📊 POSSESSION V6 RESULTS")
print("=" * 60)

print(
    f"Ball observations:       {len(ball)}"
)

valid_possession = (
    ball["possession_player"]
    .notna()
    .sum()
)

print(
    f"Possession observations: {valid_possession}"
)

coverage = (
    valid_possession /
    len(ball)
    * 100
)

print(
    f"Coverage:                {coverage:.1f}%"
)


# ============================================================
# POSSESSION BY TEAM
# ============================================================

print()
print("Possession by team:")

team_counts = (
    ball[
        ball["possession_team"].notna()
    ]
    ["possession_team"]
    .value_counts()
)


print(team_counts)


# ============================================================
# POSSESSION BY PLAYER
# ============================================================

print()
print("Possession by player:")

player_counts = (
    ball[
        ball["possession_player"].notna()
    ]
    ["possession_player"]
    .astype(int)
    .value_counts()
)


print(player_counts)


# ============================================================
# TEAM POSSESSION PERCENTAGE
# ============================================================

print()
print("Team possession percentage:")

total_team_possession = team_counts.sum()

if total_team_possession > 0:

    for team in [
        TEAM_A,
        TEAM_B
    ]:

        count = team_counts.get(
            team,
            0
        )

        percentage = (
            count /
            total_team_possession
            * 100
        )

        print(
            f"{team}: {count} frames "
            f"({percentage:.1f}%)"
        )


# ============================================================
# SEGMENTS
# ============================================================

print()
print("Possession segments:")
print(
    f"Segments: {len(segments_df)}"
)


for _, segment in segments_df.iterrows():

    print(
        f"Player {int(segment['player_id']):2d} "
        f"| {segment['team']:6s} "
        f"| {int(segment['start_frame']):3d} "
        f"-> {int(segment['end_frame']):3d}"
    )


# ============================================================
# TEAM CHANGES
# ============================================================

print()
print("Team possession changes:")

previous_team = None
team_changes = []


for _, segment in segments_df.iterrows():

    team = segment["team"]

    if (
        previous_team is not None
        and team != previous_team
    ):

        team_changes.append({
            "frame": int(
                segment["start_frame"]
            ),
            "from_team": previous_team,
            "to_team": team,
            "player": int(
                segment["player_id"]
            )
        })

    previous_team = team


if team_changes:

    for change in team_changes:

        print(
            f"Frame {change['frame']:3d}: "
            f"{change['from_team']} -> "
            f"{change['to_team']} "
            f"(Player {change['player']})"
        )

else:

    print("No team possession changes detected.")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("✅ POSSESSION V6 COMPLETE")
print("=" * 60)

print(
    f"Output: {OUTPUT_CSV}"
)

print("=" * 60)