import cv2
import numpy as np

from court_points import COURT_POINTS


# ============================================================
# COURT COORDINATE SYSTEM
# ============================================================
#
# Use the same 640x640 canonical coordinate system as the
# Roboflow training annotations.
#
# This is important: we are NOT inventing NBA coordinates yet.
# First we establish a geometrically consistent transform.
# ============================================================

TEMPLATE_WIDTH = 640
TEMPLATE_HEIGHT = 640


# ============================================================
# BUILD TEMPLATE POINTS
# ============================================================

def get_template_points():

    points = []

    for landmark_id in sorted(COURT_POINTS):

        x, y = COURT_POINTS[landmark_id]

        points.append([
            float(x),
            float(y)
        ])

    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# CALCULATE HOMOGRAPHY
# ============================================================

def calculate_homography(keypoints):

    image_points = []
    template_points = []

    for kp in keypoints:

        landmark_id = int(kp["class"])
        confidence = float(
            kp.get("confidence", 0)
        )

        if landmark_id not in COURT_POINTS:
            continue

        if confidence < 0.50:
            continue

        image_points.append([
            float(kp["x"]),
            float(kp["y"])
        ])

        x, y = COURT_POINTS[landmark_id]

        template_points.append([
            float(x),
            float(y)
        ])

    if len(image_points) < 4:

        return None, len(image_points), None

    image_points = np.array(
        image_points,
        dtype=np.float32
    )

    template_points = np.array(
        template_points,
        dtype=np.float32
    )

    H, mask = cv2.findHomography(
        image_points,
        template_points,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        return None, len(image_points), None

    inliers = int(
        mask.sum()
    ) if mask is not None else 0

    return H, len(image_points), inliers


# ============================================================
# TRANSFORM A POINT
# ============================================================

def transform_point(
    x,
    y,
    H
):

    point = np.array(
        [[[float(x), float(y)]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        H
    )

    return (
        float(transformed[0][0][0]),
        float(transformed[0][0][1])
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🏀 COURT TRANSFORM TEST")
    print("=" * 60)

    print(
        f"Template landmarks: "
        f"{len(COURT_POINTS)}"
    )

    print(
        f"Template size: "
        f"{TEMPLATE_WIDTH} x {TEMPLATE_HEIGHT}"
    )

    print()

    # --------------------------------------------------------
    # Example points
    # --------------------------------------------------------

    # These are example camera coordinates.
    # We only test that the transform machinery works.
    image_points = np.array([
        [620, 372],
        [716, 642],
        [1017, 529],
        [1071, 619],
    ], dtype=np.float32)

    # Corresponding first four template points.
    landmark_ids = sorted(COURT_POINTS)[:4]

    template_points = np.array([
        COURT_POINTS[i]
        for i in landmark_ids
    ], dtype=np.float32)

    H, mask = cv2.findHomography(
        image_points,
        template_points
    )

    if H is None:

        print("❌ Could not calculate homography")
        raise SystemExit(1)

    print("✅ Homography calculated")
    print()

    print("H =")
    print(H)

    print()

    # --------------------------------------------------------
    # Test a player point
    # --------------------------------------------------------

    test_x = 1246
    test_y = 529

    court_x, court_y = transform_point(
        test_x,
        test_y,
        H
    )

    print(
        f"Camera point: "
        f"({test_x}, {test_y})"
    )

    print(
        f"Template point: "
        f"({court_x:.2f}, {court_y:.2f})"
    )

    print()

    print("=" * 60)
    print("✅ COURT TRANSFORM TEST COMPLETE")
    print("=" * 60)
