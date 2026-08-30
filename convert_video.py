"""
convert_video.py

Basketball Tactics MVP V1
Video FPS normalization.

Rules:
    FPS > 30  -> convert to true 30 FPS
    FPS == 30 -> use original video
    FPS < 30  -> use original video

Usage:
    python convert_video.py input.mp4
"""

from pathlib import Path
import sys
import cv2


TARGET_FPS = 30.0
FPS_TOLERANCE = 0.5


def get_video_info(video_path):
    """Return FPS, frame count, width, height, and duration."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return (
        fps,
        frame_count,
        width,
        height,
        duration,
    )


def normalize_video(video_path, output_path):
    """
    Normalize video FPS for the MVP.

    Videos above 30 FPS are sampled down to 30 FPS.
    Videos at or below 30 FPS are kept unchanged.

    Returns:
        Path to the video that should be used.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    (
        fps,
        frame_count,
        width,
        height,
        duration,
    ) = get_video_info(video_path)

    print("=" * 60)
    print("🎥 VIDEO FPS CHECK")
    print("=" * 60)
    print(f"Input:       {video_path}")
    print(f"FPS:         {fps:.2f}")
    print(f"Frames:      {frame_count}")
    print(f"Resolution:  {width} x {height}")
    print(f"Duration:    {duration:.2f} sec")
    print()

    # ---------------------------------------------------------
    # CASE 1 — Already approximately 30 FPS
    # ---------------------------------------------------------

    if abs(fps - TARGET_FPS) < FPS_TOLERANCE:

        print("✅ Video is already approximately 30 FPS.")
        print("Using original video.")
        print("=" * 60)

        return video_path

    # ---------------------------------------------------------
    # CASE 2 — Below 30 FPS
    # ---------------------------------------------------------

    if fps < TARGET_FPS:

        print(
            f"⚠️ Input FPS is below "
            f"{TARGET_FPS:.0f}."
        )
        print("Keeping original FPS.")
        print("=" * 60)

        return video_path

    # ---------------------------------------------------------
    # CASE 3 — Above 30 FPS
    # ---------------------------------------------------------

    print(
        f"🔄 Converting "
        f"{fps:.2f} FPS → "
        f"{TARGET_FPS:.0f} FPS"
    )
    print()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        TARGET_FPS,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    # ---------------------------------------------------------
    # True FPS downsampling
    # ---------------------------------------------------------

    frame_index = 0
    written_frames = 0

    frame_interval = fps / TARGET_FPS

    next_output_frame = 0.0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index >= next_output_frame:

            writer.write(frame)

            written_frames += 1

            next_output_frame += frame_interval

        frame_index += 1

    cap.release()
    writer.release()

    print("✅ Conversion complete.")
    print(f"Output:      {output_path}")
    print(f"Output FPS:  {TARGET_FPS:.0f}")
    print(f"Input frames:{frame_index}")
    print(f"Output frames: {written_frames}")
    print("=" * 60)

    return output_path


def main():

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "  python convert_video.py "
            "<input_video>"
        )
        print()
        print("Example:")
        print(
            "  python convert_video.py "
            "data/videos/game.mp4"
        )

        sys.exit(1)

    input_video = Path(
        sys.argv[1]
    )

    if not input_video.exists():

        print(
            f"❌ File not found: "
            f"{input_video}"
        )

        sys.exit(1)

    output_video = Path(
        "data/processed/video_30fps.mp4"
    )

    # Remove stale normalized video.
    # This prevents the pipeline from accidentally
    # using a previous video's output.

    if output_video.exists():

        output_video.unlink()

        print(
            f"🧹 Removed old normalized video: "
            f"{output_video}"
        )
        print()

    result = normalize_video(
        input_video,
        output_video,
    )

    print()
    print(
        f"🎯 Pipeline video: {result}"
    )


if __name__ == "__main__":
    main()