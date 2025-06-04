from ultralytics import YOLO
import os
import cv2
import numpy as np
from glob import glob
import torch

# Try importing box_iou from different possible locations
try:
    from ultralytics.utils.ops import box_iou
except ImportError:
    try:
        from ultralytics.yolo.utils.ops import box_iou
    except ImportError:
        try:
            from ultralytics.utils.metrics import box_iou
        except ImportError:
            # Fallback: use torchvision's box_iou
            from torchvision.ops import box_iou

from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.metrics import ap_per_class

# Paths
model_path = '/Users/adil_zhiyenbayev/adil_code/helmet_detection/Safety-Detection-YOLOv8/models/ppe.pt'
data_yaml = '/path/to/your/data.yaml'
images_dir = '/Users/adil_zhiyenbayev/adil_code/dataset_resized_640/test/images'
labels_dir = '/Users/adil_zhiyenbayev/adil_code/dataset_resized_640/test/labels'

# Model class indices
MODEL_HARDHAT_IDX = 0
MODEL_NO_HARDHAT_IDX = 2

# Dataset class mapping
DATASET_CLASS_MAP = {
    MODEL_HARDHAT_IDX: 0,         # "Hardhat" -> 0
    MODEL_NO_HARDHAT_IDX: 1       # "NO-Hardhat" -> 1
}

# Load model
model = YOLO(model_path)

# Get all test images
image_paths = glob(os.path.join(images_dir, '*.jpg'))

# Initialize metrics
stats = []
iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()

for img_path in image_paths:
    # Load image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Run inference
    results = model(img)
    
    # Process predictions
    for r in results:
        boxes = r.boxes
        scores = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        # Filter and remap classes
        mask = np.isin(cls, [0, 2])  # Only keep Hardhat and NO-Hardhat
        if not mask.any():
            continue
            
        boxes = boxes[mask]
        scores = scores[mask]
        cls = cls[mask]  # No remapping needed
        
        # Convert boxes to xyxy format
        pred_boxes = boxes.xyxy.cpu().numpy()
        
        # Load ground truth
        label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        gt_boxes = []
        gt_cls = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    c = int(data[0])
                    if c in [0, 1]:  # Only consider Hardhat and NO-Hardhat in your dataset
                        # Convert normalized xywh to pixel xyxy
                        x, y, w, h = map(float, data[1:5])
                        x1 = (x - w/2) * width
                        y1 = (y - h/2) * height
                        x2 = (x + w/2) * width
                        y2 = (y + h/2) * height
                        gt_boxes.append([x1, y1, x2, y2])
                        # Remap: 0 stays 0 (Hardhat), 1 (NO-Hardhat in dataset) becomes 2 (NO-Hardhat in model)
                        if c == 1:
                            gt_cls.append(2)
                        else:
                            gt_cls.append(c)
        
        gt_boxes = np.array(gt_boxes)
        gt_cls = np.array(gt_cls)
        
        # Compute metrics
        correct = np.zeros((len(pred_boxes), niou))
        if len(gt_boxes):
            # Compute iou between pred and ground truth
            iou = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))
            
            # Assign predictions to ground truth objects
            for i, j in enumerate(iou.argmax(1)):
                if iou[i, j] > iouv[0]:  # if over threshold
                    if gt_cls[j] == cls[i]:  # if classes match
                        correct[i] = iou[i, j] >= iouv
        
        # Append statistics
        stats.append((correct, scores, cls, gt_cls))

        print("Predicted classes in this image:", cls)
        print("Ground truth classes in this image:", gt_cls)

# Compute metrics
stats = [np.concatenate(x, 0) for x in zip(*stats)]
if len(stats) and stats[0].any():
    precision, recall, ap, f1, ap_class, *rest = ap_per_class(*stats, plot=False, names=['Hardhat', 'NO-Hardhat'])
    print("ap shape:", ap.shape)
    if ap.ndim == 2:
        ap50, ap_mean = ap[:, 0], ap.mean(1)
    else:
        ap50, ap_mean = ap[0], ap.mean()
    mp, mr, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap_mean.mean()
    
    # Convert scalar ap50 and ap_mean to lists if they're scalars
    ap50 = [ap50] if np.isscalar(ap50) else ap50
    ap_mean = [ap_mean] if np.isscalar(ap_mean) else ap_mean
    
    # Ensure AP arrays are length 2 (for both classes)
    if len(ap50) < 2:
        ap50 = list(ap50) + [0.0] * (2 - len(ap50))
    if len(ap_mean) < 2:
        ap_mean = list(ap_mean) + [0.0] * (2 - len(ap_mean))
    
    # Print results
    print(f'\nResults:')
    print(f'mAP@0.5: {map50:.3f}')
    print(f'mAP@0.5:0.95: {map:.3f}')
    print('\nPer-class results:')
    print('Class      AP@0.5  AP@0.5:0.95')
    class_names = ['Hardhat', 'NO-Hardhat']
    
    for i in range(2):  # Always iterate over both classes
        cname = class_names[i]
        print(f'{cname:10s} {ap50[i]:.3f}  {ap_mean[i]:.3f}')