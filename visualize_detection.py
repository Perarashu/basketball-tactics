import os
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("❌ ROBOFLOW_API_KEY not found")
    exit()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

print("🚀 Running detection...")

result = client.infer(
    "data/frame.jpg",
    model_id="basketball-players-fy4c2/25"
)

image = cv2.imread("data/frame.jpg")

if image is None:
    print("❌ Could not load frame.jpg")
    exit()

for prediction in result["predictions"]:

    x = prediction["x"]
    y = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]

    confidence = prediction["confidence"]
    class_name = prediction["class"]

    # Convert center coordinates to corner coordinates
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/detection.jpg"

cv2.imwrite(output_path, image)

print(f"✅ Detection image saved to {output_path}")