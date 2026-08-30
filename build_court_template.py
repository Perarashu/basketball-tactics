import json
from pathlib import Path

# ============================================================
# BUILD COURT LANDMARK TEMPLATE
# ============================================================

SCHEMA_PATH = (
    "data/basketball_court_schema/valid/_annotations.coco.json"
)

OUTPUT_PATH = "court_points.py"


print("=" * 60)
print("🏀 BUILDING COURT LANDMARK TEMPLATE")
print("=" * 60)


# ============================================================
# LOAD COCO DATA
# ============================================================

with open(SCHEMA_PATH, "r") as f:
    data = json.load(f)


# ============================================================
# FIND COURT CATEGORY
# ============================================================

court_category = None

for category in data["categories"]:
    if category["name"] == "court":
        court_category = category
        break

if court_category is None:
    raise RuntimeError("❌ Could not find 'court' category")


keypoint_names = court_category["keypoints"]

print(f"Landmarks: {len(keypoint_names)}")


# ============================================================
# COLLECT LANDMARK POSITIONS
# ============================================================

landmark_samples = {}

for name in keypoint_names:
    landmark_id = int(name)

    landmark_samples[landmark_id] = []


for annotation in data["annotations"]:

    keypoints = annotation.get("keypoints", [])

    if not keypoints:
        continue

    for index, name in enumerate(keypoint_names):

        base = index * 3

        if base + 2 >= len(keypoints):
            continue

        x = keypoints[base]
        y = keypoints[base + 1]
        visibility = keypoints[base + 2]

        # COCO visibility:
        # 0 = not labeled
        # 1 = labeled but not visible
        # 2 = visible

        if visibility <= 0:
            continue

        landmark_id = int(name)

        landmark_samples[landmark_id].append(
            (float(x), float(y))
        )


# ============================================================
# CALCULATE AVERAGE POSITION
# ============================================================

COURT_POINTS = {}

for landmark_id in sorted(landmark_samples):

    samples = landmark_samples[landmark_id]

    if not samples:
        continue

    avg_x = sum(x for x, y in samples) / len(samples)
    avg_y = sum(y for x, y in samples) / len(samples)

    COURT_POINTS[landmark_id] = (
        avg_x,
        avg_y
    )


# ============================================================
# PRINT
# ============================================================

print()
print("============================================================")
print("📐 COURT LANDMARK TEMPLATE")
print("============================================================")

for landmark_id in sorted(COURT_POINTS):

    x, y = COURT_POINTS[landmark_id]

    print(
        f"{landmark_id:02d}: "
        f"({x:.2f}, {y:.2f})"
    )


# ============================================================
# WRITE PYTHON FILE
# ============================================================

with open(OUTPUT_PATH, "w") as f:

    f.write("# Auto-generated court landmark template\n\n")

    f.write("COURT_POINTS = {\n")

    for landmark_id in sorted(COURT_POINTS):

        x, y = COURT_POINTS[landmark_id]

        f.write(
            f"    {landmark_id}: "
            f"({x:.6f}, {y:.6f}),\n"
        )

    f.write("}\n")


print()
print("=" * 60)
print("✅ TEMPLATE CREATED")
print("=" * 60)
print(f"Saved: {OUTPUT_PATH}")
print(f"Landmarks included: {len(COURT_POINTS)}")
print("=" * 60)