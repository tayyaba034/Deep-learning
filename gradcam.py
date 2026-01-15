"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Model Interpretability
Visualize which parts of the image the model focuses on
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize
                       If None, uses the last convolutional layer
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name} for Grad-CAM")
        
        # Create gradient model
        self.grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed)
            pred_index: Target class index (None for predicted class)
            eps: Small constant for numerical stability
        
        Returns:
            Heatmap array
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get conv outputs and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Use predicted class if not specified
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the class output
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of class output w.r.t. conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get conv outputs
        conv_outputs = conv_outputs[0]
        
        # Weight the conv outputs by the gradients
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU (only positive influences)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (0-255 or 0-1 range)
            alpha: Transparency of overlay
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image
        """
        # Ensure original image is in 0-255 range
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to 0-255 range
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid
    
    def visualize(self, image, original_image=None, save_path=None):
        """
        Generate and visualize Grad-CAM
        
        Args:
            image: Preprocessed image for model
            original_image: Original image for visualization
            save_path: Path to save visualization
        
        Returns:
            Figure object
        """
        # Use preprocessed image if original not provided
        if original_image is None:
            original_image = image.copy()
        
        # Compute heatmap
        heatmap = self.compute_heatmap(image)
        
        # Overlay heatmap
        overlaid = self.overlay_heatmap(heatmap, original_image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if original_image.max() <= 1.0:
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlaid)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        
        return fig


def visualize_multiple_images(model, images, original_images=None, 
                              layer_name=None, save_path=None):
    """
    Visualize Grad-CAM for multiple images
    
    Args:
        model: Keras model
        images: List of preprocessed images
        original_images: List of original images (optional)
        layer_name: Layer name for Grad-CAM
        save_path: Path to save visualization
    
    Returns:
        Figure object
    """
    grad_cam = GradCAM(model, layer_name)
    
    if original_images is None:
        original_images = images
    
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, orig_img) in enumerate(zip(images, original_images)):
        # Compute heatmap
        heatmap = grad_cam.compute_heatmap(img)
        overlaid = grad_cam.overlay_heatmap(heatmap, orig_img)
        
        # Plot
        if orig_img.max() <= 1.0:
            axes[idx, 0].imshow(orig_img)
        else:
            axes[idx, 0].imshow(orig_img.astype(np.uint8))
        axes[idx, 0].set_title(f'Image {idx+1}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title(f'Heatmap {idx+1}', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlaid)
        axes[idx, 2].set_title(f'Overlay {idx+1}', fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-image Grad-CAM visualization saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Grad-CAM visualization module loaded successfully!")
    print("\nUsage example:")
    print("""
    from utils.gradcam import GradCAM
    
    # Create Grad-CAM object
    grad_cam = GradCAM(model, layer_name='conv2d_15')  # or None for auto-detect
    
    # Visualize single image
    fig = grad_cam.visualize(preprocessed_image, original_image, save_path='gradcam.png')
    
    # Compute heatmap only
    heatmap = grad_cam.compute_heatmap(preprocessed_image)
    """)
