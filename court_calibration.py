"""
court_calibration.py

Converts the Roboflow court-template coordinates into
a standard basketball court coordinate system.

Output coordinate system:

    X = 0 ... 94 feet
    Y = 0 ... 50 feet

Origin:
    top-left corner of the court

This gives downstream systems a stable physical coordinate
system for heatmaps, movement analysis and tactical events.
"""

import cv2
import numpy as np

from court_points import COURT_POINTS


# ============================================================
# STANDARD BASKETBALL COURT
# ============================================================

COURT_WIDTH_FT = 94.0
COURT_HEIGHT_FT = 50.0


# ============================================================
# ROBOTFLOW TEMPLATE LANDMARKS
# ============================================================
#
# These four landmarks represent the four outer corners
# of the court in the Roboflow template.
#
# Based on the geometry in court_points.py:
#
#       1 ---------------- 15
#       |                   |
#       |                   |
#       |                   |
#       8 ---------------- 41
#
# ============================================================

CALIBRATION_LANDMARKS = {
    1: (0.0, 0.0),
    15: (COURT_WIDTH_FT, 0.0),
    41: (COURT_WIDTH_FT, COURT_HEIGHT_FT),
    8: (0.0, COURT_HEIGHT_FT),
}


# ============================================================
# BUILD CALIBRATION HOMOGRAPHY
# ============================================================

def build_calibration_homography():
    """
    Build a homography from the Roboflow template coordinates
    to a real 94 x 50 ft basketball court.
    """

    template_points = []
    court_points = []

    for landmark_id, target in CALIBRATION_LANDMARKS.items():

        if landmark_id not in COURT_POINTS:
            raise RuntimeError(
                f"Calibration landmark {landmark_id} "
                f"is missing from COURT_POINTS."
            )

        template_x, template_y = COURT_POINTS[landmark_id]

        court_x, court_y = target

        template_points.append([
            float(template_x),
            float(template_y),
        ])

        court_points.append([
            float(court_x),
            float(court_y),
        ])

    template_points = np.array(
        template_points,
        dtype=np.float32,
    )

    court_points = np.array(
        court_points,
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(
        template_points,
        court_points,
    )

    return H


# ============================================================
# GLOBAL CALIBRATION MATRIX
# ============================================================

CALIBRATION_H = build_calibration_homography()


# ============================================================
# TRANSFORM TEMPLATE POINT
# ============================================================

def template_to_court(x, y):
    """
    Convert a Roboflow template coordinate into
    real basketball court coordinates.

    Returns:
        court_x, court_y

    Units:
        feet
    """

    point = np.array(
        [[[float(x), float(y)]]],
        dtype=np.float32,
    )

    transformed = cv2.perspectiveTransform(
        point,
        CALIBRATION_H,
    )

    court_x = float(
        transformed[0, 0, 0]
    )

    court_y = float(
        transformed[0, 0, 1]
    )

    return court_x, court_y


# ============================================================
# CLAMP TO COURT
# ============================================================

def clamp_to_court(x, y):
    """
    Keep a point inside the physical court boundaries.
    """

    x = max(
        0.0,
        min(COURT_WIDTH_FT, x)
    )

    y = max(
        0.0,
        min(COURT_HEIGHT_FT, y)
    )

    return x, y


# ============================================================
# TRANSFORM + CLAMP
# ============================================================

def template_to_court_clamped(x, y):
    """
    Convert template coordinates to physical court
    coordinates and keep the result inside the court.
    """

    court_x, court_y = template_to_court(
        x,
        y,
    )

    return clamp_to_court(
        court_x,
        court_y,
    )


# ============================================================
# DEBUG
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🏀 COURT CALIBRATION TEST")
    print("=" * 60)

    print()
    print("Court size:")
    print(
        f"  Width:  {COURT_WIDTH_FT} ft"
    )
    print(
        f"  Height: {COURT_HEIGHT_FT} ft"
    )

    print()
    print("Calibration landmarks:")

    for landmark_id, target in CALIBRATION_LANDMARKS.items():

        source = COURT_POINTS[landmark_id]

        print(
            f"  ID {landmark_id:2d}: "
            f"template=({source[0]:7.2f}, {source[1]:7.2f}) "
            f"→ court=({target[0]:5.1f}, {target[1]:5.1f})"
        )

    print()
    print("Calibration matrix:")
    print(CALIBRATION_H)

    print()
    print("Testing calibration...")

    for landmark_id, target in CALIBRATION_LANDMARKS.items():

        template_x, template_y = COURT_POINTS[
            landmark_id
        ]

        court_x, court_y = template_to_court(
            template_x,
            template_y,
        )

        error = np.sqrt(
            (court_x - target[0]) ** 2
            +
            (court_y - target[1]) ** 2
        )

        print(
            f"  ID {landmark_id:2d}: "
            f"({court_x:6.2f}, {court_y:6.2f}) "
            f"error={error:.4f} ft"
        )

    print()
    print("=" * 60)
    print("✅ COURT CALIBRATION READY")
    print("=" * 60)