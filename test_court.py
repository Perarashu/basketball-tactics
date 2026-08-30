import cv2
import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# ==========================================
# READ FRAME
# ==========================================

cap = cv2.VideoCapture(
    "data/videos/game_30fps.mp4"
)

success, frame = cap.read()

cap.release()

if not success:
    print("❌ Could not read video")
    exit()


# ==========================================
# COURT INFERENCE
# ==========================================

print("🚀 Detecting basketball court...")

result = client.infer(
    frame,
    model_id="basketball-court-detection-2/22"
)

print("✅ Court detection complete!")


# ==========================================
# SELECT BEST COURT
# ==========================================

predictions = result["predictions"]

if len(predictions) == 0:

    print("❌ No court detected")
    exit()


best_prediction = max(
    predictions,
    key=lambda p: p["confidence"]
)

print(
    f"Best court confidence: "
    f"{best_prediction['confidence']:.3f}"
)


# ==========================================
# DRAW KEYPOINTS
# ==========================================

keypoints = best_prediction.get(
    "keypoints",
    []
)

print(
    f"Keypoints detected: "
    f"{len(keypoints)}"
)

for kp in keypoints:

    x = int(kp["x"])
    y = int(kp["y"])

    confidence = kp["confidence"]

    # Ignore very uncertain points
    if confidence < 0.5:
        continue


    # Draw point
    cv2.circle(
        frame,
        (x, y),
        8,
        (0, 255, 255),
        -1
    )


    # Draw keypoint name
    label = (
        f"{kp['class']} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )


# ==========================================
# SAVE IMAGE
# ==========================================

output_path = (
    "outputs/court_keypoints.jpg"
)

cv2.imwrite(
    output_path,
    frame
)

print()
print("===================================")
print("✅ COURT VISUALIZATION COMPLETE")
print("===================================")
print(
    f"Saved to: {output_path}"
)
print("===================================")