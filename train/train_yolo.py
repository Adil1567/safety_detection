import os
os.environ["WANDB_MODE"] = "disabled"
import yaml
import numpy as np
from ultralytics import YOLO

class CFG:
    DEBUG = False
    FRACTION = 0.10 if DEBUG else 1.0
    SEED = 88
    CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
               'NO-Safety Vest', 'Person', 'Safety Cone',
               'Safety Vest', 'machinery', 'vehicle']
    NUM_CLASSES_TO_TRAIN = len(CLASSES)
    EPOCHS = 3 if DEBUG else 120
    BATCH_SIZE = 16
    BASE_MODEL = 'yolov8m'
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
    EXP_NAME = f'ppe_css_{EPOCHS}_epochs'
    OPTIMIZER = 'auto'
    LR = 1e-3
    LR_FACTOR = 0.01
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.025
    PATIENCE = 25
    PROFILE = False
    LABEL_SMOOTHING = 0.0    
    CUSTOM_DATASET_DIR = '/home/adilz/safety_detection/3/css-data'
    OUTPUT_DIR = './'

def prepare_data_yaml(cfg):
    dict_file = {
        'train': os.path.join(cfg.CUSTOM_DATASET_DIR, 'train'),
        'val': os.path.join(cfg.CUSTOM_DATASET_DIR, 'valid'),
        'test': os.path.join(cfg.CUSTOM_DATASET_DIR, 'test'),
        'nc': cfg.NUM_CLASSES_TO_TRAIN,
        'names': cfg.CLASSES
    }
    yaml_path = os.path.join(cfg.OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w+') as file:
        yaml.dump(dict_file, file)
    return yaml_path

def get_image_properties(image_path):
    import cv2
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
    yaml_path = prepare_data_yaml(cfg)
    # Use a sample image to get image size
    example_image_path = os.path.join(cfg.CUSTOM_DATASET_DIR, 'train/images', os.listdir(os.path.join(cfg.CUSTOM_DATASET_DIR, 'train/images'))[0])
    img_properties = get_image_properties(example_image_path)
    # model = YOLO(cfg.BASE_MODEL_WEIGHTS)
    model = YOLO('yolov8m.pt')
    model.train(
        data=yaml_path,
        task='detect',
        imgsz=(img_properties['height'], img_properties['width']),
        epochs=cfg.EPOCHS,
        batch=cfg.BATCH_SIZE,
        optimizer=cfg.OPTIMIZER,
        lr0=cfg.LR,
        lrf=cfg.LR_FACTOR,
        weight_decay=cfg.WEIGHT_DECAY,
        dropout=cfg.DROPOUT,
        fraction=cfg.FRACTION,
        patience=cfg.PATIENCE,
        profile=cfg.PROFILE,
        label_smoothing=cfg.LABEL_SMOOTHING,
        name=f'{cfg.BASE_MODEL}_{cfg.EXP_NAME}',
        seed=cfg.SEED,
        val=True,
        amp=True,
        classes=[0,1,2,3,4,5,7],
        exist_ok=True,
        resume=False,
        device=[1],  # Change as needed
        verbose=False,
    )

if __name__ == "__main__":
    main() 
# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")
