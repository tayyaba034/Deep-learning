"""
Prediction/Inference Script
Implemented in PyTorch
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from cnn_pytorch import SimpleCNN, ImprovedCNN

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict digits from images using PyTorch model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'improved'],
                       help='Model architecture')
    parser.add_argument('--show_probabilities', action='store_true',
                       help='Show prediction probabilities for all classes')
    parser.add_argument('--invert', action='store_true',
                       help='Invert image colors (useful for black digits on white background)')
    
    return parser.parse_args()


def load_and_preprocess_image(image_path, invert=False):
    """
    Load and preprocess image for prediction
    
    Args:
        image_path: Path to image file
        invert: Force invert colors
        
    Returns:
        Preprocessed image array
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img).astype('float32') / 255.0
    
    # Auto-invert: If average of corners is bright, assume black-on-white
    corners = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    if (invert) or (np.mean(corners) > 0.5):
        img_array = 1.0 - img_array
    
    # Reshape for PyTorch: (1, 1, 28, 28)
    img_array = img_array.reshape(1, 1, 28, 28)
    
    # Apply same normalization as training
    mean, std = 0.1307, 0.3081
    img_array = (img_array - mean) / std
    
    return img_array


def load_pytorch_model(model_path, model_type='simple'):
    """Load PyTorch model"""
    # Create model
    if model_type == 'simple':
        model = SimpleCNN(num_classes=10)
    else:
        model = ImprovedCNN(num_classes=10)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if checkpoint is a state_dict or a full checkpoint with 'model_state_dict' key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model


def predict(model, image):
    """Make prediction using PyTorch model"""
    image_tensor = torch.FloatTensor(image)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].numpy()


def plot_prediction(image, predicted_class, confidence, probabilities, show_probs=False):
    """
    Visualize prediction
    
    Args:
        image: Original image array
        predicted_class: Predicted digit
        confidence: Confidence of prediction
        probabilities: Probability distribution
        show_probs: Whether to show probability bar chart
    """
    if show_probs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    
    # Display image
    img_display = image.squeeze()
    # Un-normalize for display
    mean, std = 0.1307, 0.3081
    img_display = img_display * std + mean
    
    ax1.imshow(img_display, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%',
                 fontsize=14, fontweight='bold')
    
    # Show probability distribution
    if show_probs:
        classes = list(range(10))
        bars = ax2.bar(classes, probabilities * 100, color='skyblue', edgecolor='navy')
        bars[predicted_class].set_color('green')
        
        ax2.set_xlabel('Digit', fontsize=12)
        ax2.set_ylabel('Probability (%)', fontsize=12)
        ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xticks(classes)
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1:  # Only show labels for probabilities > 1%
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def main():
    """Main prediction function"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    print("=" * 70)
    print("MNIST DIGIT PREDICTION (PyTorch)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print("=" * 70)
    
    # Load and preprocess image
    image = load_and_preprocess_image(args.image_path, invert=args.invert)
    
    # Load model and make prediction
    print(f"\nLoading PyTorch model...")
    model = load_pytorch_model(args.model_path, args.model_type)
    predicted_class, confidence, probabilities = predict(model, image)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Predicted Digit: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
    if args.show_probabilities:
        print("\nProbability Distribution:")
        print("-" * 70)
        for i, prob in enumerate(probabilities):
            bar = '█' * int(prob * 50)
            print(f"Digit {i}: {prob*100:6.2f}% {bar}")
        print("=" * 70)
    
    # Visualize prediction
    print("\nGenerating visualization...")
    fig = plot_prediction(image, predicted_class, confidence, probabilities, 
                         show_probs=args.show_probabilities)
    
    # Save visualization
    output_path = args.image_path.replace('.', '_prediction.')
    if not output_path.endswith(('.png', '.jpg', '.jpeg')):
        output_path = args.image_path + '_prediction.png'
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
