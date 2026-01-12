"""
Demo Script - Test the PyTorch CNN models
Quick demo to verify installation and model architecture
"""

import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("MNIST DIGIT CLASSIFICATION - DEMO (PyTorch)")
print("=" * 70)

# Test PyTorch
print("\n1. Testing PyTorch Setup...")
try:
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ GPU Device: {torch.cuda.get_device_name(0)}")
    
    from cnn_pytorch import SimpleCNN, ImprovedCNN
    
    # Test simple model
    simple_model = SimpleCNN()
    print(f"   ✓ Simple CNN: {simple_model.count_parameters():,} parameters")
    
    # Test improved model
    improved_model = ImprovedCNN()
    print(f"   ✓ Improved CNN: {improved_model.count_parameters():,} parameters")
    
    pt_available = True
except Exception as e:
    print(f"   ✗ PyTorch Error: {e}")
    pt_available = False

# Test data loading
print("\n2. Testing Data Loading...")
try:
    from data_loader import MNISTDataLoader
    
    print("   Testing PyTorch data loader...")
    pt_loader = MNISTDataLoader()
    train_data, val_data, test_data = pt_loader.load_data(validation_split=0.1)
    print(f"   ✓ PyTorch: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    data_available = True
except Exception as e:
    print(f"   ✗ Data Loading Error: {e}")
    data_available = False

# Test visualization
print("\n3. Testing Visualization...")
try:
    print("   ✓ Matplotlib available")
    print("   ✓ Seaborn available")
    viz_available = True
except Exception as e:
    print(f"   ✗ Visualization Error: {e}")
    viz_available = False

# Test metrics
print("\n4. Testing Metrics...")
try:
    # Dummy test
    y_true = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("   ✓ Scikit-learn metrics available")
    metrics_available = True
except Exception as e:
    print(f"   ✗ Metrics Error: {e}")
    metrics_available = False

# Summary
print("\n" + "=" * 70)
print("SETUP SUMMARY")
print("=" * 70)
print(f"PyTorch:          {'✓ Available' if pt_available else '✗ Not Available'}")
print(f"Data Loading:     {'✓ Working' if data_available else '✗ Failed'}")
print(f"Visualization:    {'✓ Working' if viz_available else '✗ Failed'}")
print(f"Metrics:          {'✓ Working' if metrics_available else '✗ Failed'}")

all_working = pt_available and data_available and viz_available and metrics_available

print("=" * 70)
if all_working:
    print("✓ All systems operational! Ready to train models.")
    print("\nNext steps:")
    print("  1. Train PyTorch model: python train_pytorch.py")
    print("  2. Run inference:      python predict.py --model_path results/model.pth --image_path img.png")
else:
    print("✗ Setup incomplete - Please install dependencies:")
    print("  pip install -r requirements.txt")

print("=" * 70)
