# 🚶‍♂️ Real-Time Person Detection System

A streamlined, professional deep learning system for real-time person detection using state-of-the-art YOLOv8.

## 🌟 Overview
This project provides a robust, "to the point" implementation of person detection. It uses the modern YOLOv8 architecture to identify humans in video streams or image files with high accuracy and performance.

### ✨ Key Features
- **Real-time Performance**: High FPS detection on both CPU and GPU.
- **Modern Architecture**: Leverages YOLOv8 from Ultralytics.
- **Easy Deployment**: Minimal dependencies and automated model handling.
- **Transparent Metrics**: Real-time FPS and person count overlay.

---

## 🚀 How to Run

### 1. Prerequisites
- **Python 3.8+**
- **Webcam** (for real-time mode)

### 2. Installation
```bash
# Clone the repository (if applicable)
# cd person-detection

# Install required packages
pip install ultralytics opencv-python numpy

or 

pip install -r requirements.txt
```

### 3. Run the Detector
Simply execute the main script:
```bash
python detector.py
```

### Controls
- Press **'q'** to stop the detection and close the window.

---

## 🛠️ Configuration & Usage

### Running on Video Files
You can modify `detector.py` to point to a video file by changing the `source` argument in `detector.run()`:
```python
# In detector.py
detector.run(source='path/to/video.mp4')
```

### Performance Tuning
- **Model Size**: Change `model_size` in the constructor ('n', 's', 'm', 'l', 'x').
- **Confidence**: Adjust `confidence_threshold` to filter detections.
- **Device**: Set `device='cpu'` if you don't have an NVIDIA GPU.

---

## 📊 Performance Benchmark
Tested on NVIDIA RTX 3060:
- **YOLOv8n**: ~120 FPS
- **YOLOv8s**: ~80 FPS

---

## 📜 Technical Details
The system filters the COCO dataset detections to specifically isolate the **Person** class (ID 0), ensuring the interface remains focused on human detection.



