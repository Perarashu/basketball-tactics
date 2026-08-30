import os
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

print("🚀 Sending frame to Roboflow...")

result = client.infer(
    "data/frame.jpg",
    model_id="basketball-players-fy4c2/25"
)

print("✅ Inference completed!")
print(result)