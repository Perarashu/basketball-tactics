"""
pipeline.py

Basketball Tactics MVP V1 Pipeline

Pipeline:
    1. Normalize video FPS
    2. Extract player coordinates
    3. Assign teams
    4. Transform player coordinates to court coordinates
    5. Estimate possession
    6. Build tactical events

Pass detection is intentionally NOT part of V1.
"""

from pathlib import Path
import subprocess
import sys


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "outputs"
PROCESSED_DIR = ROOT / "data" / "processed"


# ============================================================
# RUN SCRIPT
# ============================================================

def run_script(script, *args):
    """
    Run another pipeline stage and stop immediately if it fails.
    """

    script_path = ROOT / script

    if not script_path.exists():
        raise RuntimeError(
            f"Missing pipeline script: {script}"
        )

    command = [
        sys.executable,
        str(script_path),
        *[str(arg) for arg in args],
    ]

    print()
    print("=" * 70)
    print(f"▶ RUNNING: {script}")
    print("=" * 70)

    subprocess.run(
        command,
        check=True,
    )


# ============================================================
# CHECK FILE
# ============================================================

def check_file(path):
    """
    Verify that an expected output exists.
    """

    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"Expected output was not created:\n{path}"
        )

    print(f"✅ {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # ---------------------------------------------------------
    # ARGUMENT CHECK
    # ---------------------------------------------------------

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print("  python pipeline.py <input_video>")
        print()
        print("Example:")
        print("  python pipeline.py data/videos/game.mp4")
        print()

        sys.exit(1)

    input_video = Path(sys.argv[1])

    # Convert relative paths to absolute paths
    if not input_video.is_absolute():
        input_video = ROOT / input_video

    input_video = input_video.resolve()

    # ---------------------------------------------------------
    # INPUT CHECK
    # ---------------------------------------------------------

    if not input_video.exists():

        raise RuntimeError(
            f"Input video not found:\n{input_video}"
        )

    # ---------------------------------------------------------
    # CREATE DIRECTORIES
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print("=" * 70)
    print("🏀 BASKETBALL TACTICS — MVP V1")
    print("=" * 70)

    print()
    print(f"Input video: {input_video}")

    # =========================================================
    # STEP 1 — FPS NORMALIZATION
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 1 — VIDEO FPS NORMALIZATION")
    print("=" * 70)

    run_script(
        "convert_video.py",
        input_video,
    )

    normalized_video = (
        PROCESSED_DIR / "video_30fps.mp4"
    )

    # ---------------------------------------------------------
    # Determine which video to use
    # ---------------------------------------------------------
    #
    # convert_video.py creates video_30fps.mp4 when the input
    # is above 30 FPS.
    #
    # If the input is already <=30 FPS, convert_video.py keeps
    # the original video.
    #
    # Therefore:
    #   - use normalized video if it exists
    #   - otherwise use original input
    #

    if normalized_video.exists():

        normalized_video = normalized_video.resolve()

    else:

        normalized_video = input_video

    print()
    print(f"Using video:")
    print(f"  {normalized_video}")

    # =========================================================
    # STEP 2 — PLAYER COORDINATE EXTRACTION
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 2 — PLAYER COORDINATE EXTRACTION")
    print("=" * 70)

    run_script(
        "extract_coordinates.py",
        normalized_video,
    )

    check_file(
        OUTPUT_DIR / "player_coordinates_v2.csv"
    )

    # =========================================================
    # STEP 3 — TEAM ASSIGNMENT
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 3 — TEAM ASSIGNMENT")
    print("=" * 70)

    run_script(
        "team_assignment.py",
    )

    check_file(
        OUTPUT_DIR / "player_team_assignments.csv"
    )

    check_file(
        OUTPUT_DIR / "player_coordinates_with_teams.csv"
    )

    # =========================================================
    # STEP 4 — COURT COORDINATES
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 4 — COURT COORDINATE TRANSFORMATION")
    print("=" * 70)

    run_script(
        "transform_player_coordinates.py",
    )

    check_file(
        OUTPUT_DIR / "player_coordinates_with_teams.csv"
    )

    # =========================================================
    # STEP 5 — POSSESSION
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 5 — POSSESSION ESTIMATION")
    print("=" * 70)

    run_script(
        "possession_v6.py",
    )

    check_file(
        OUTPUT_DIR / "possession_v6.csv"
    )

    # =========================================================
    # STEP 6 — TACTICAL EVENTS
    # =========================================================

    print()
    print("=" * 70)
    print("STEP 6 — TACTICAL EVENT BUILDING")
    print("=" * 70)

    run_script(
        "build_tactical_events.py",
    )

    check_file(
        OUTPUT_DIR / "tactical_events.csv"
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print()
    print("=" * 70)
    print("✅ MVP V1 PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print("Input:")
    print(f"  {input_video}")

    print()
    print("Video used:")
    print(f"  {normalized_video}")

    print()
    print("Generated outputs:")
    print()

    generated_outputs = [
        "player_coordinates_v2.csv",
        "player_team_assignments.csv",
        "player_coordinates_with_teams.csv",
        "possession_v6.csv",
        "tactical_events.csv",
    ]

    for filename in generated_outputs:

        print(
            f"  outputs/{filename}"
        )

    print()
    print("Pipeline stages:")
    print("  ✅ FPS normalization")
    print("  ✅ Player detection + tracking")
    print("  ✅ Team assignment")
    print("  ✅ Court coordinate transformation")
    print("  ✅ Possession estimation")
    print("  ✅ Tactical event extraction")

    print()
    print("Next layer:")
    print("  Streamlit + Qwen2.5-7B-Instruct")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()