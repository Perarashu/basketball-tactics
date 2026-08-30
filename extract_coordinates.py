import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


MODEL_ID = "basketball-players-fy4c2/25"


def extract_player_coordinates(
    video_path,
    output_csv="outputs/player_coordinates_v2.csv",
    max_seconds=10,
):
    """
    Detect and track players in a video using Roboflow + ByteTrack.

    Parameters
    ----------
    video_path : str
        Path to the input video.

    output_csv : str
        Where player coordinates will be saved.

    max_seconds : int | None
        Maximum number of seconds to process.
        Use None to process the complete video.

    Returns
    -------
    str
        Path to the generated player-coordinate CSV.
    """

    video_path = Path(video_path)

    # ---------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------

    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not found. "
            "Add ROBOFLOW_API_KEY to your .env file."
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # ROBOFLOW
    # ---------------------------------------------------------

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # ---------------------------------------------------------
    # TRACKER
    # ---------------------------------------------------------

    tracker = sv.ByteTrack()

    # ---------------------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video FPS."
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ---------------------------------------------------------
    # FRAME LIMIT
    # ---------------------------------------------------------

    if max_seconds is None:

        max_frames = total_frames

    else:

        max_frames = min(
            total_frames,
            int(fps * max_seconds),
        )

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    print(
        "============================================================"
    )
    print("🏀 PLAYER COORDINATE EXTRACTION")
    print(
        "============================================================"
    )
    print(f"Video:        {video_path}")
    print(f"FPS:          {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Processing:   {max_frames} frames")
    print("")

    coordinates = []

    frame_number = 0

    # ---------------------------------------------------------
    # PROCESS VIDEO
    # ---------------------------------------------------------

    while frame_number < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------
        # ROBOFLOW INFERENCE
        # -----------------------------------------------------

        try:

            result = client.infer(
                frame,
                model_id=MODEL_ID,
            )

        except Exception as exc:

            print(
                f"⚠️ Roboflow inference failed at frame "
                f"{frame_number}: {exc}"
            )

            frame_number += 1
            continue

        predictions = result.get(
            "predictions",
            [],
        )

        # -----------------------------------------------------
        # CONVERT PREDICTIONS TO DETECTIONS
        # -----------------------------------------------------

        xyxy = []
        confidence = []
        class_id = []

        for prediction in predictions:

            x = prediction.get("x")
            y = prediction.get("y")
            width = prediction.get("width")
            height = prediction.get("height")

            if (
                x is None
                or y is None
                or width is None
                or height is None
            ):
                continue

            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2

            xyxy.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            confidence.append(
                float(
                    prediction.get(
                        "confidence",
                        0,
                    )
                )
            )

            class_id.append(
                int(
                    prediction.get(
                        "class_id",
                        0,
                    )
                )
            )

        # -----------------------------------------------------
        # CREATE SUPERVISION DETECTIONS
        # -----------------------------------------------------

        if xyxy:

            detections = sv.Detections(
                xyxy=np.array(
                    xyxy,
                    dtype=float,
                ),
                confidence=np.array(
                    confidence,
                    dtype=float,
                ),
                class_id=np.array(
                    class_id,
                    dtype=int,
                ),
            )

        else:

            detections = sv.Detections.empty()

        # -----------------------------------------------------
        # BYTE TRACK
        # -----------------------------------------------------

        tracked = tracker.update_with_detections(
            detections
        )

        # -----------------------------------------------------
        # SAVE PLAYER COORDINATES
        # -----------------------------------------------------

        for i in range(len(tracked)):

            if tracked.tracker_id is None:
                continue

            tracker_id = tracked.tracker_id[i]

            if tracker_id is None:
                continue

            x1, y1, x2, y2 = tracked.xyxy[i]

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            coordinates.append(
                {
                    "frame": frame_number,
                    "player_id": int(
                        tracker_id
                    ),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float(
                        center_x
                    ),
                    "center_y": float(
                        center_y
                    ),
                }
            )

        frame_number += 1

        # -----------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------

        if frame_number % 30 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{max_frames} frames..."
            )

    cap.release()

    # ---------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------

    df = pd.DataFrame(
        coordinates
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    print("")
    print(
        "============================================================"
    )
    print(
        "📊 PLAYER EXTRACTION COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        f"Frames processed: {frame_number}"
    )
    print(
        f"Player rows:      {len(df)}"
    )
    print(
        f"Output:           {output_csv}"
    )
    print(
        "============================================================"
    )

    return output_csv


# =============================================================
# COMMAND-LINE ENTRY POINT
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        input_video = sys.argv[1]
    else:
        input_video = "data/videos/game_30fps.mp4"

    extract_player_coordinates(
        input_video,
        max_seconds=None,
    )