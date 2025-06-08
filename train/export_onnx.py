import os
os.environ["WANDB_MODE"] = "disabled"
from ultralytics import YOLO
import cv2

class CFG:
    BASE_MODEL = 'yolov8s'
    EXP_NAME = f'ppe_css_80_epochs'  # Should match your training config
    CUSTOM_DATASET_DIR = '/home/adilz/safety_detection/3/css-data'
    MODEL_WEIGHTS = '/home/adilz/safety_detection/runs/detect/yolov8s_ppe_css_80_epochs/weights/best.pt'
    OUTPUT_ONNX = 'best2.onnx'  # Output ONNX file name

def get_image_properties(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")
    properties = {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": img.dtype,
    }
    return properties

def main():
    cfg = CFG()
    # Use a sample image to get image size
    example_image_path = os.path.join(cfg.CUSTOM_DATASET_DIR, 'train/images', os.listdir(os.path.join(cfg.CUSTOM_DATASET_DIR, 'train/images'))[0])
    img_properties = get_image_properties(example_image_path)
    # Load model
    model = YOLO(cfg.MODEL_WEIGHTS)
    # Export to ONNX
    print("Exporting model to ONNX format...")
    model.export(
        format='onnx',
        imgsz=(img_properties['height'], img_properties['width']),
        half=False,
        int8=False,
        simplify=True,
        nms=False,
        dynamic=False,
        optimize=False,
        opset=None
    )
    print(f"Model exported to {cfg.OUTPUT_ONNX}")

if __name__ == "__main__":
    main() 