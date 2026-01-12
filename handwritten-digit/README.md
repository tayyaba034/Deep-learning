# MNIST Handwritten Digit Classification Project (PyTorch)

A comprehensive deep learning project for classifying handwritten digits (0-9) using Convolutional Neural Networks (CNNs) on the MNIST dataset, implemented exclusively in **PyTorch**.

---

## 📚 Table of Contents
1. [Project Overview](#-project-overview)
2. [✨ Key Features](#-key-features)
3. [🚀 Quick Start](#-quick-start)
4. [📂 Project Structure](#-project-structure)
5. [🏗️ Technical Architecture](#-technical-architecture)
6. [📖 Usage Guide](#-usage-guide)
7. [📊 Performance & Results](#-performance--results)
8. [🚀 Advanced Topics](#-advanced-topics)
9. [🔧 Troubleshooting](#-troubleshooting)
10. [🛠️ Requirements](#-requirements)
11. [🤝 Contributing & License](#-contributing--license)

---

## 🎯 Project Overview

This project demonstrates:
- **CNN Architecture Design**: Building and comparing different CNN architectures in PyTorch.
- **Best Practices**: Using proper validation, callbacks, and evaluation metrics.
- **Production Ready**: Creating modular, well-documented, and reusable code.

### The MNIST Dataset
- 70,000 grayscale images of handwritten digits (0-9).
- Training set: 60,000 images | Test set: 10,000 images.
- Image size: 28×28 pixels.

---

## ✨ Key Features

- ✅ **PyTorch Implementation**: Complete, optimized training and inference using PyTorch.
- ✅ **Multiple Architectures**: Simple and Improved CNN models.
- ✅ **Comprehensive Evaluation**: Metrics (Accuracy, Precision, Recall, F1), confusion matrices, and error analysis.
- ✅ **Visualization**: Training history curves, sample predictions, and misclassification analysis.
- ✅ **Easy to Use**: Clean command-line scripts for training and inference.
- ✅ **Educational**: Perfect for learning CNN fundamentals with PyTorch.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python demo.py
```

### 3. Train a Model
```bash
python train_pytorch.py --model simple --epochs 10
```

### 4. Make Predictions
```bash
python predict.py \
    --model_path results/best_model_simple_pt.pth \
    --image_path test_digit.png \
    --show_probabilities
```

---

## 📂 Project Structure

```
mnist_digit_classification/
├── README.md                     # Main project documentation (PyTorch Only)
├── requirements.txt              # Python dependencies
│
├── cnn_pytorch.py                # PyTorch CNN implementations
│
├── data_loader.py                # PyTorch data loading
├── visualization.py              # Plotting and visualization
├── metrics.py                    # Evaluation metrics
│
├── train_pytorch.py              # PyTorch training script
├── predict.py                    # Inference script
├── demo.py                       # Setup verification
│
└── results/                      # Output directory (created during training)
    ├── best_model_*.pth          # Trained models
    ├── metrics_*.txt             # Performance metrics
    └── *.png                     # Visualization plots
```

---

## 🏗️ Technical Architecture

### CNN Architectures

#### 1. Simple CNN
- **Architecture**: 2 Conv blocks → Flatten → Dense layer with Dropout.
- **Parameters**: ~1.2 million.
- **Best for**: Quick training, understanding basic concepts.

#### 2. Improved CNN
- **Architecture**: 3 Conv blocks with Batch Normalization → Dense layers with Dropout.
- **Parameters**: ~2.5 million.
- **Best for**: Higher accuracy (99%+), production-grade performance.

### Training Strategy
- **Optimizer**: Adam (Adaptive learning rate).
- **Loss**: CrossEntropyLoss.
- **Regularization**: Dropout, Batch Normalization, Early Stopping.
- **Preprocessing**: Pixel normalization, reshaping, and transform-based augmentation.

---

## 📖 Usage Guide

### Training Options
`train_pytorch.py` supports these arguments:
- `--model`: `simple` or `improved` (Default: `simple`).
- `--epochs`: Number of training epochs (Default: `10`).
- `--batch_size`: Size of training batches (Default: `128`).
- `--learning_rate`: Initial learning rate (Default: `0.001`).
- `--device`: `auto`, `cuda`, or `cpu` (Default: `auto`).
- `--save_dir`: Directory to save results (Default: `results`).

### Detailed Prediction
```bash
python predict.py \
    --model_path {path_to_model} \
    --image_path {path_to_image} \
    --model_type {simple|improved} \
    --show_probabilities
```

---

## 📊 Performance & Results

Expected results on the MNIST test set using PyTorch:

| Model | Parameters | Test Accuracy | Training Time (GPU) |
|-------|-----------|---------------|---------------------|
| Simple CNN | ~1.2M | 98.8 - 99.2% | ~2-3 minutes |
| Improved CNN | ~2.5M | 99.2 - 99.5% | ~5-7 minutes |

### Visualizing Results
After training, check the `results/` folder for:
- `training_history.png`: Loss and accuracy curves.
- `confusion_matrix.png`: Prediction error distribution.
- `sample_predictions.png`: Visual verification of model output.

---

## 🚀 Advanced Topics

### 1. Data Augmentation
To improve generalization, use `torchvision.transforms` to add rotations, shifts, and normalization to training data.

### 2. Hyperparameter Tuning
Experiment with different learning rates [0.001, 0.0001], optimizers [Adam, SGD, RMSprop], and batch sizes [64, 128, 256].

### 3. Model Compression
Explore techniques like Quantization or TorchScript to reduce model size for mobile/edge deployment.

---

## 🔧 Troubleshooting

- **CUDA Out of Memory**: Reduce `--batch_size` to 64 or 32.
- **Low Accuracy**: Ensure data transforms are consistent; train for more epochs (e.g., `--epochs 20`).
- **Import Errors**: Run `pip install -r requirements.txt --upgrade` to ensure all libraries are correct.
- **Slow Training**: Ensure your device is set to `cuda` if you have an NVIDIA GPU.

---

## 🛠️ Requirements

### System Requirements
- **Python**: 3.8 or higher.
- **RAM**: 4GB minimum (8GB recommended).
- **GPU**: Recommended but not required (CUDA 11.x+).

### Core Libraries
- PyTorch 2.0+
- Torchvision 0.15+
- NumPy, Matplotlib, Seaborn
- scikit-learn, Pillow

---

## 🤝 Contributing & License

Feel free to fork this project and experiment with new architectures!

**License**: MIT License - Free for educational and commercial use.

---

**Happy Learning! 🎓**  
*Ready to start? Run `python demo.py` to begin!*
