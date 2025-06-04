import pandas as pd
import numpy as np
from mean_average_precision import MetricBuilder

# Paths to your CSVs
gt_path = '/Users/adil_zhiyenbayev/adil_code/helmet_detection/Safety-Detection-YOLOv8/images/archive/labels/ground_truth_test_dataset_resized640.csv'
pred_path = '/Users/adil_zhiyenbayev/adil_code/helmet_detection/Safety-Detection-YOLOv8/images/archive/labels/predictions_test_resided640.csv'

# Load CSVs
df_gt = pd.read_csv(gt_path)
df_pred = pd.read_csv(pred_path)

# Map ground truth class names to match predictions
# name_map = {"With Helmet": "Helmet", "Without Helmet": "Without_Helmet"}
# df_gt['class'] = df_gt['class'].map(name_map).fillna(df_gt['class'])

# Map class names to integer IDs
class_map = {'Helmet': 0, 'Without_Helmet': 1}
df_gt['class_id'] = df_gt['class'].map(class_map)
df_pred['class_id'] = df_pred['class'].map(class_map)

# Add 'difficult' and 'crowd' columns to ground truth (set to 0)
df_gt['difficult'] = 0
df_gt['crowd'] = 0

# If your class is 'head' in the ground truth:
df_gt['class'] = df_gt['class'].replace({'head': 'Helmet'})

# Prepare numpy arrays
gt_arr = df_gt[['x1', 'y1', 'x2', 'y2', 'class_id', 'difficult', 'crowd']].to_numpy(dtype=np.float32)
pred_arr = df_pred[['x1', 'y1', 'x2', 'y2', 'class_id', 'confidence']].to_numpy(dtype=np.float32)

# Compute mAP
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=2)
metric_fn.add(pred_arr, gt_arr)
print(metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05)))

