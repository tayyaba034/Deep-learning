"""
Visualization Utilities for MNIST Digit Classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_sample_images(images, labels, predictions=None, num_samples=10):
    """
    Plot sample images with labels and predictions
    
    Args:
        images: Array of images
        labels: True labels
        predictions: Predicted labels (optional)
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Get image
        img = images[i].squeeze().numpy() if hasattr(images[i], 'numpy') else images[i].squeeze()
        true_label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Set title
        if predictions is not None:
            pred_label = predictions[i].item() if hasattr(predictions[i], 'item') else predictions[i]
            
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        else:
            axes[i].set_title(f'Label: {true_label}', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Training history dict
    """
    # Extract from dict (PyTorch)
    train_acc = history['train_accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs = range(1, len(train_acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names=None, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_per_class_accuracy(cm, class_names=None):
    """
    Plot per-class accuracy from confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, per_class_acc * 100, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_misclassified_samples(images, true_labels, pred_labels, num_samples=10):
    """
    Plot misclassified samples
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        num_samples: Number of samples to display
    """
    # Find misclassified samples
    true_classes = true_labels
    pred_classes = pred_labels
    
    misclassified_idx = np.where(true_classes != pred_classes)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return None
    
    # Select random misclassified samples
    num_samples = min(num_samples, len(misclassified_idx))
    sample_idx = np.random.choice(misclassified_idx, num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(sample_idx):
        img = images[idx].squeeze()
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'True: {true_classes[idx]}\nPred: {pred_classes[idx]}', 
                         color='red', fontsize=10)
    
    # Hide empty subplots
    for i in range(num_samples, 10):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_learning_rate_schedule(learning_rates, epochs):
    """
    Plot learning rate schedule
    
    Args:
        learning_rates: List of learning rates
        epochs: List of epoch numbers
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, learning_rates, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results_dict):
    """
    Compare multiple models performance
    
    Args:
        results_dict: Dictionary with model names as keys and (train_acc, val_acc, test_acc) as values
    """
    models = list(results_dict.keys())
    train_accs = [results_dict[m][0] for m in models]
    val_accs = [results_dict[m][1] for m in models]
    test_accs = [results_dict[m][2] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, train_accs, width, label='Train Accuracy', alpha=0.8)
    bars2 = ax.bar(x, val_accs, width, label='Validation Accuracy', alpha=0.8)
    bars3 = ax.bar(x + width, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def save_all_plots(figures, save_dir='results'):
    """
    Save all generated figures
    
    Args:
        figures: Dictionary of {filename: figure} pairs
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for filename, fig in figures.items():
        if fig is not None:
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
    
    plt.close('all')


if __name__ == "__main__":
    # Test visualization functions with dummy data
    print("Testing visualization functions...")
    
    # Create dummy data
    dummy_images = np.random.rand(10, 28, 28, 1)
    dummy_labels = np.eye(10)[:10]
    dummy_predictions = np.eye(10)[:10]
    
    # Test sample images plot
    fig1 = plot_sample_images(dummy_images, dummy_labels, dummy_predictions)
    
    # Test training history plot
    dummy_history = {
        'train_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.75, 0.82, 0.87, 0.89, 0.91],
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.75, 0.58, 0.42, 0.32, 0.25]
    }
    fig2 = plot_training_history(dummy_history)
    
    # Test confusion matrix
    dummy_cm = np.random.randint(0, 100, size=(10, 10))
    fig3 = plot_confusion_matrix(dummy_cm)
    
    print("Visualization tests completed!")
    plt.show()
