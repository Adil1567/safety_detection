import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat

# Add both classes to the safe globals
torch.serialization.add_safe_globals([
    DetectionModel,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.container.ModuleList,
    Conv,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    C2f,
    Bottleneck,
    SPPF,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    Concat
])

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from ultralyticsplus import YOLO
import numpy as np
import cv2
import base64
# import torch
from typing import Optional

from datetime import datetime
import sqlite3
import os

app = FastAPI()

# Load your model once at startup
model = YOLO('/Users/adil_zhiyenbayev/adil_code/helmet_detection/Safety-Detection-YOLOv8/models/models--keremberke--yolov8n-hard-hat-detection/snapshots/287bafa2feb311ee45d21f9e9b33315ff6ff955d/best.pt')
# model = YOLO("/Users/adil_zhiyenbayev/adil_code/hard-hat-detection/yolov8n.pt")
# model = YOLO("/Users/adil_zhiyenbayev/adil_code/hard-hat-detection/ultralytics/runs/detect/yolov8n_custom_default/weights/best.pt")
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000
classNames = ['Helmet', 'Without_Helmet']

# New multi-class model
model_multiclass = YOLO('/Users/adil_zhiyenbayev/adil_code/helmet_detection/models/best_yolo8m.pt')
model_multiclass.overrides['conf'] = 0.25
model_multiclass.overrides['iou'] = 0.45
model_multiclass.overrides['agnostic_nms'] = False
model_multiclass.overrides['max_det'] = 1000
classNames_multiclass = ['Helmet', 'Mask', 'Without_Helmet', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
# classNames = ['Helmet', 'Mask', 'Without_Helmet', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#                   'Safety Vest', 'machinery', 'vehicle']


def save_violation(image_path: str, no_helmet: int, no_mask: int, no_vest: int):
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            timestamp TEXT,
            image_path TEXT,
            no_helmet INTEGER,
            no_gloves INTEGER,
            no_goggles INTEGER
        )
    """)
    c.execute("INSERT INTO violations VALUES (?, ?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), image_path, no_helmet, no_mask, no_vest))
    conn.commit()
    conn.close()





@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file as bytes
    contents = await file.read()
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    # Run inference
    results = model.predict(img)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls]
            detections.append({
                "class": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })
    return {"detections": detections}

@app.post("/predict-both/")
async def predict_both(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Invalid image received")
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    
    # Log the image dimensions
    print(f"Image dimensions: {img.shape}")
    
    # Run inference
    results = model.predict(img)
    detections = []
    cropped_images = []

    # Create a copy of the original image for drawing bounding boxes
    img_with_boxes = img.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls]
            color = (0, 255, 0) if label == 'Helmet' else (0, 0, 255)
            
            # Draw bounding boxes and labels on the copied image
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
            detections.append({
                "class": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

            # Crop the detection if it's "Without_Helmet" (from the original image)
            if label == "Without_Helmet":
                cropped_img = img[y1:y2, x1:x2]  # Crop from the original image
                # Resize the cropped image to a minimum size
                min_size = (100, 100)
                if cropped_img.shape[0] < min_size[0] or cropped_img.shape[1] < min_size[1]:
                    cropped_img = cv2.resize(cropped_img, min_size)
                # Log the dimensions of the cropped image
                print(f"Cropped image dimensions: {cropped_img.shape}")
                cropped_filename = f"without_helmet_{len(cropped_images)}.jpg"
                cv2.imwrite(cropped_filename, cropped_img)
                cropped_images.append(cropped_filename)

    # Save the full image with detections
    os.makedirs("images", exist_ok=True)

    # Generate a unique timestamped filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    output_path = f"images/{filename}"
    
    cv2.imwrite(output_path, img_with_boxes)

    # Return the detections, full image URL, and cropped image filenames
    return {
        "detections": detections,
        "image_url": f"/get-image/{output_path}",
        "cropped_images": cropped_images  # Return only the filenames
    }

@app.post("/predict-both-multiclass/")
async def predict_both_multiclass(
    file: UploadFile = File(...),
    selected_classes: Optional[str] = Form(None)  # Accept as comma-separated string
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Invalid image received")
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    
    print(f"Image dimensions: {img.shape}")
    results = model_multiclass.predict(img)
    detections = []
    cropped_images = []
    img_with_boxes = img.copy()

    # Parse selected_classes
    if selected_classes:
        selected_classes_set = set([c.strip() for c in selected_classes.split(",")])
    else:
        selected_classes_set = None  # No filtering

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames_multiclass[cls]

            # Only process if label is in selected_classes (or no filter)
            if (selected_classes_set is None) or (label in selected_classes_set):
                color = (0, 0, 255) if label in ['Without_Helmet', 'NO-Mask', 'NO-Safety Vest'] else (0, 255, 0)
                
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_with_boxes, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                detections.append({
                    "class": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })
                # Crop if label is in the selected set and is a crop class
                if label in ["Without_Helmet", "NO-Safety Vest", 'NO-Mask']:
                    cropped_img = img[y1:y2, x1:x2]
                    min_size = (100, 100)
                    if cropped_img.shape[0] < min_size[0] or cropped_img.shape[1] < min_size[1]:
                        cropped_img = cv2.resize(cropped_img, min_size)
                    cropped_filename = f"{label.lower()}_{len(cropped_images)}.jpg"
                    cv2.imwrite(cropped_filename, cropped_img)
                    cropped_images.append(cropped_filename)

    os.makedirs("images", exist_ok=True)

    # Generate a unique timestamped filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    output_path = f"images/{filename}"
    
    cv2.imwrite(output_path, img_with_boxes)
    
    # working with DB
    no_helmet = any(d["class"] == "Without_Helmet" for d in detections)
    no_mask = any(d["class"] == "NO-Mask" for d in detections)
    no_vest = any(d["class"] == "NO-Safety Vest" for d in detections)
    
    save_violation(output_path, int(no_helmet), int(no_mask), int(no_vest))
    
    return {
        "detections": detections,
        "image_url": f"/get-image/{output_path}",
        "cropped_images": cropped_images
    }

@app.get("/get-image/{file_path:path}")
async def get_image(file_path: str):
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/get-violations/")
async def get_violations():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("SELECT * FROM violations ORDER BY timestamp DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()
    # Convert to JSON
    keys = ["timestamp", "image_path", "no_helmet", "no_mask", "no_vest"]
    result = [dict(zip(keys, row)) for row in rows]
    return JSONResponse(content=result)
