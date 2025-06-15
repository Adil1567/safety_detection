# Model Files

## Available Models

### 1. Multiclass PPE Detection Model
- Download from: [Google Drive](https://drive.google.com/file/d/1IR8ouKL9e0McJ-cq7TgmGZcWslqgEQMo/view?usp=sharing_)
- File to download: `ppe.pt` (83.6 MB)
- Classes:
  - Hardhat (class 0)
  - Mask (class 1)
  - NO-Hardhat (class 2)
  - NO-Mask (class 3)
  - NO-Safety Vest (class 4)
  - Person (class 5)
  - Safety Cone (class 6)
  - Safety Vest (class 7)
  - machinery (class 8)
  - vehicle (class 9)

### 2. Binary Hardhat Detection Model
- Download from: [Hugging Face Hub](https://huggingface.co/keremberke/yolov8n-hard-hat-detection)
- Classes:
  - Hardhat
  - NO-Hardhat

## Setup Instructions

1. Download the desired model:
   - For multiclass detection: Download `best_yolov8m.pt` from the Google Drive link
   - For binary detection: The model will be automatically downloaded from Hugging Face Hub to `~/.cache/huggingface/hub/`

2. Place the downloaded model file(s) in this directory (`Safety-Detection-YOLOv8/models/`)

Note: The code is configured to use the multiclass model by default. If you want to use the binary model, you'll need to modify the model loading configuration in the code.
