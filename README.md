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

# PPE Detection System

A real-time Personal Protective Equipment (PPE) detection system using YOLOv8, FastAPI, and Streamlit. This system can detect various safety violations including missing helmets, masks, and safety vests in real-time.

## 🚀 Features

- **Multi-class Detection**: Detects multiple PPE items and violations:
  - Helmets (with/without)
  - Face masks (with/without)
  - Safety vests (with/without)
  - Additional classes: Person, Safety Cone, Machinery, Vehicle

- **Multiple Interfaces**:
  - FastAPI backend for model inference
  - Gradio interface for easy model testing
  - Streamlit dashboard for violation monitoring

- **Real-time Monitoring**:
  - Live violation detection
  - Automatic image capture of violations
  - Violation logging with timestamps
  - SQLite database for violation storage

- **Dashboard Features**:
  - Real-time violation statistics
  - Historical data visualization
  - Image gallery of violations
  - Export functionality for violation logs

## 🛠️ Technical Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Backend**: FastAPI
- **Frontend**: 
  - Gradio (for model testing)
  - Streamlit (for dashboard)
- **Database**: SQLite
- **Image Processing**: OpenCV
- **Additional Libraries**: NumPy, Pandas

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppe-detection.git
cd ppe-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 models:
- Place your trained models in the `Safety-Detection-YOLOv8/models/` directory
- Required models:
  - `best.pt` (2-class model)
  - `best_yolo8m.pt` (multi-class model)

## 💻 Usage

### 1. Start the FastAPI Server
```bash
uvicorn main:app --reload
```

### 2. Launch the Gradio Interface
```bash
python gradio.py
```

### 3. Start the Streamlit Dashboard
```bash
streamlit run streamlit.py
```

## 📊 API Endpoints

- `/predict/`: Basic detection endpoint
- `/predict-both/`: Detection with image saving
- `/predict-both-multiclass/`: Multi-class detection
- `/get-image/{file_path}`: Retrieve saved images
- `/get-violations/`: Get violation history

## 📈 Dashboard Features

- Real-time violation monitoring
- Historical data visualization
- Violation type statistics
- Image gallery of violations
- Export functionality for violation logs

## 🔒 Database Schema

The system uses SQLite with the following schema:
```sql
CREATE TABLE violations (
    timestamp TEXT,
    image_path TEXT,
    no_helmet INTEGER,
    no_mask INTEGER,
    no_vest INTEGER
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors

- Adil Zhiyenbayev

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- FastAPI team
- Streamlit team
- Gradio team

