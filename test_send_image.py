import requests
from datetime import datetime

# Path to your local image
IMAGE_PATH = "trade.jpg"  # Make sure this image exists

# URL of your local Gradio FastAPI server
URL = "http://localhost:8000/predict-both-multiclass/"

# Optional: specify which classes to detect
selected_classes = ",".join(["Without_Helmet", "NO-Mask", "NO-Safety Vest"])

with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    data = {"selected_classes": selected_classes}
    response = requests.post(URL, files=files, data=data)

if response.ok:
    print("✅ Inference successful.")
    result = response.json()
    print("Detections:")
    for det in result["detections"]:
        print(f" - {det['class']} ({det['confidence']:.2f})")
else:
    print("❌ Failed:", response.status_code, response.text)
