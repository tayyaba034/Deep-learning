"""
Main Training Script for Cat vs Dog Classification
Complete end-to-end training pipeline
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import build_custom_cnn, build_efficientnet_transfer
from augmentation import AdvancedAugmentation, MixupCutmix, create_data_generator
from training import TrainingConfig, TrainingPipeline


def download_sample_data():
    """
    Download sample cat and dog images for demonstration
    Note: For full dataset, use Kaggle's Dogs vs Cats dataset
    """
    print("For full training, download the Kaggle Dogs vs Cats dataset:")
    print("https://www.kaggle.com/c/dogs-vs-cats/data")
    print("\nFor quick testing, this script will create dummy data.")
    return None


def prepare_data(data_dir, img_size=224, validation_split=0.2, test_split=0.1):
    """
    Prepare data for training
    
    Args:
        data_dir: Directory containing cat and dog images
        img_size: Target image size
        validation_split: Validation set ratio
        test_split: Test set ratio
    
    Returns:
        (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    print("Preparing data...")
    
    # Get image paths
    cat_images = glob(os.path.join(data_dir, 'cat*.jpg')) + \
                 glob(os.path.join(data_dir, 'cats', '*.jpg'))
    dog_images = glob(os.path.join(data_dir, 'dog*.jpg')) + \
                 glob(os.path.join(data_dir, 'dogs', '*.jpg'))
    
    if len(cat_images) == 0 or len(dog_images) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check the directory.")
    
    # Create labels (0 for cat, 1 for dog)
    image_paths = cat_images + dog_images
    labels = [0] * len(cat_images) + [1] * len(dog_images)
    
    print(f"Found {len(cat_images)} cat images and {len(dog_images)} dog images")
    
    # Split data
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=validation_split/(1-test_split), 
        random_state=42, 
        stratify=train_val_labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)


def train_model(data_dir, model_type='custom', config=None, output_dir='outputs'):
    """
    Train the model
    
    Args:
        data_dir: Directory containing training data
        model_type: 'custom' or 'efficientnet'
        config: TrainingConfig instance
        output_dir: Directory to save outputs
    
    Returns:
        Trained model and history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    if config is None:
        config = TrainingConfig()
    
    # Save config
    config.save(os.path.join(output_dir, 'config.json'))
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_data(data_dir, config.img_size)
    
    # Create augmentation
    train_aug = AdvancedAugmentation(
        img_size=config.img_size, 
        augmentation_level=config.augmentation_level
    )
    val_aug = AdvancedAugmentation(img_size=config.img_size, augmentation_level='light')
    
    # Create mixup/cutmix
    mixup_cutmix = None
    if config.use_mixup_cutmix:
        mixup_cutmix = MixupCutmix(alpha=config.mixup_alpha)
    
    # Create data generators
    train_generator = create_data_generator(
        train_paths, train_labels, 
        batch_size=config.batch_size,
        augmentation=train_aug,
        shuffle=True,
        mixup_cutmix=mixup_cutmix
    )
    
    val_generator = create_data_generator(
        val_paths, val_labels,
        batch_size=config.batch_size,
        augmentation=val_aug,
        shuffle=False
    )
    
    test_generator = create_data_generator(
        test_paths, test_labels,
        batch_size=config.batch_size,
        augmentation=val_aug,
        shuffle=False
    )
    
    # Calculate steps
    steps_per_epoch = len(train_paths) // config.batch_size
    validation_steps = len(val_paths) // config.batch_size
    test_steps = len(test_paths) // config.batch_size
    
    # Build model
    print(f"\nBuilding {model_type} model...")
    if model_type == 'custom':
        model = build_custom_cnn(input_shape=(config.img_size, config.img_size, 3))
    elif model_type == 'efficientnet':
        model = build_efficientnet_transfer(input_shape=(config.img_size, config.img_size, 3))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.summary()
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.compile_model(model)
    
    # Train model
    model_save_path = os.path.join(output_dir, f'best_{model_type}_model.keras')
    history = pipeline.train(
        train_generator=train_generator,
        val_generator=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        model_save_path=model_save_path
    )
    
    # Plot training history
    pipeline.plot_training_history(
        save_path=os.path.join(output_dir, f'{model_type}_training_history.png')
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = pipeline.evaluate(test_generator, test_steps)
    
    # Save test results
    import json
    with open(os.path.join(output_dir, f'{model_type}_test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    return model, history, test_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Cat vs Dog Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing cat and dog images')
    parser.add_argument('--model_type', type=str, default='custom',
                       choices=['custom', 'efficientnet'],
                       help='Model architecture to use')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.img_size = args.img_size
    config.initial_lr = args.learning_rate
    
    # Train model
    model, history, test_results = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        config=config,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    print("="*80)


if __name__ == "__main__":
    # For testing without command line arguments
    if len(sys.argv) == 1:
        print("Cat vs Dog Classification Training Script")
        print("\nUsage:")
        print("  python main_train.py --data_dir /path/to/data --model_type custom")
        print("\nArguments:")
        print("  --data_dir: Directory containing cat and dog images")
        print("  --model_type: Model architecture (custom or efficientnet)")
        print("  --output_dir: Output directory (default: outputs)")
        print("  --epochs: Number of epochs (default: 50)")
        print("  --batch_size: Batch size (default: 32)")
        print("  --img_size: Image size (default: 224)")
        print("  --learning_rate: Learning rate (default: 1e-3)")
    else:
        main()
