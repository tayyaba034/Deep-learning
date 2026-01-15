"""
Interactive Web Interface for Cat vs Dog Classification
Features:
- Real-time image upload and prediction
- Grad-CAM visualization
- Confidence scores
- Batch prediction support
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import architectures

from gradcam import GradCAM


class CatDogPredictor:
    """Cat vs Dog predictor with visualization"""
    
    def __init__(self, model_path):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
        """
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.img_size = 224
        self.grad_cam = GradCAM(self.model)
        print("Model loaded successfully!")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def predict(self, image, show_gradcam=True):
        """
        Predict class and generate visualization
        
        Args:
            image: Input image
            show_gradcam: Whether to show Grad-CAM visualization
        
        Returns:
            (prediction_text, confidence_dict, visualization_image)
        """
        # Store original image
        original_image = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0][0]
        
        # Interpret prediction
        if prediction > 0.5:
            pred_class = "Dog"
            confidence = prediction
        else:
            pred_class = "Cat"
            confidence = 1 - prediction
        
        # Create confidence dictionary for display
        confidence_dict = {
            "Cat": float(1 - prediction),
            "Dog": float(prediction)
        }
        
        # Create result text
        result_text = f"Prediction: **{pred_class}** (Confidence: {confidence:.2%})"
        
        # Generate Grad-CAM visualization if requested
        if show_gradcam:
            heatmap = self.grad_cam.compute_heatmap(processed_image)
            
            # Resize original image if needed
            if original_image.shape[:2] != (self.img_size, self.img_size):
                original_resized = cv2.resize(original_image, (self.img_size, self.img_size))
            else:
                original_resized = original_image
            
            # Overlay heatmap
            visualization = self.grad_cam.overlay_heatmap(heatmap, original_resized)
        else:
            visualization = original_image
        
        return result_text, confidence_dict, visualization


def create_interface(model_path):
    """
    Create Gradio interface
    
    Args:
        model_path: Path to trained model
    
    Returns:
        Gradio interface
    """
    predictor = CatDogPredictor(model_path)
    
    def predict_wrapper(image, show_gradcam):
        """Wrapper function for Gradio"""
        if image is None:
            return "Please upload an image", {}, None
        
        result_text, confidence_dict, visualization = predictor.predict(image, show_gradcam)
        return result_text, confidence_dict, visualization
    
    # Create interface
    interface = gr.Interface(
        fn=predict_wrapper,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Checkbox(value=True, label="Show Grad-CAM Visualization")
        ],
        outputs=[
            gr.Markdown(label="Prediction"),
            gr.Label(label="Confidence Scores", num_top_classes=2),
            gr.Image(label="Visualization")
        ],
        title="🐱🐶 Cat vs Dog Classifier with AI Explainability",
        description="""
        Upload an image of a cat or dog, and the AI will classify it!
        
        **Features:**
        - Custom CNN with attention mechanisms
        - Grad-CAM visualization showing what the model focuses on
        - Real-time predictions with confidence scores
        
        **Tips:**
        - Use clear, well-lit images for best results
        - Images will be automatically resized to 224x224
        - Enable Grad-CAM to see which parts of the image influenced the prediction
        """,
        examples=[
            # Add example images if available
        ],
        theme=gr.themes.Soft(),
        analytics_enabled=False
    )
    
    return interface


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Cat vs Dog Classifier Web Interface')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.keras)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the interface on')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Create and launch interface
    interface = create_interface(args.model_path)
    
    print("\n" + "="*80)
    print("Launching Cat vs Dog Classifier Web Interface...")
    print("="*80)
    
    interface.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    # For testing without command line arguments
    if len(sys.argv) == 1:
        print("Cat vs Dog Classifier Web Interface")
        print("\nUsage:")
        print("  python web_interface.py --model_path /path/to/model.keras")
        print("\nArguments:")
        print("  --model_path: Path to trained model file")
        print("  --port: Port number (default: 7860)")
        print("  --share: Create a public link")
        print("\nExample:")
        print("  python web_interface.py --model_path outputs/best_custom_model.keras --share")
    else:
        main()
