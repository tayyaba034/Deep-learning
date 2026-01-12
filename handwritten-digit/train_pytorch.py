"""
Training Script for PyTorch CNN
MNIST Handwritten Digit Classification
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_pytorch import SimpleCNN, ImprovedCNN, ModelTrainer
from data_loader import MNISTDataLoader
from visualization import (
    plot_training_history, plot_confusion_matrix, plot_per_class_accuracy,
    plot_sample_images, plot_misclassified_samples, save_all_plots
)
from metrics import (
    calculate_confusion_matrix, calculate_metrics, print_metrics_summary,
    print_classification_report, analyze_errors, print_error_analysis,
    save_metrics_to_file
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CNN on MNIST Dataset (PyTorch)')
    
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'improved'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer to use')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def get_predictions(model, data_loader, device):
    """Get predictions for entire dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("MNIST HANDWRITTEN DIGIT CLASSIFICATION")
    print("Framework: PyTorch")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer}")
    print("=" * 70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    data_loader = MNISTDataLoader()
    train_dataset, val_dataset, test_dataset = data_loader.load_data(
        validation_split=args.validation_split
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Build model
    print(f"\nBuilding {args.model} CNN model...")
    if args.model == 'simple':
        model = SimpleCNN(num_classes=10)
    else:
        model = ImprovedCNN(num_classes=10)
    
    model = model.to(device)
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {model.count_parameters():,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    model_path = os.path.join(args.save_dir, f'best_model_{args.model}_pt.pth')
    
    # Train model
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, model_path)
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    
    # Load best model
    print(f"\nLoading best model from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_true, y_pred_probs = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics_summary(metrics)
    
    # Classification report
    print_classification_report(y_true, y_pred)
    
    # Error analysis
    analysis = analyze_errors(y_true, y_pred)
    print_error_analysis(analysis)
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, f'metrics_{args.model}_pt.txt')
    save_metrics_to_file(metrics, metrics_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    figures = {}
    
    # Training history
    figures['training_history_pt.png'] = plot_training_history(history)
    
    # Confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    figures['confusion_matrix_pt.png'] = plot_confusion_matrix(cm, normalize=False)
    figures['confusion_matrix_normalized_pt.png'] = plot_confusion_matrix(cm, normalize=True)
    
    # Per-class accuracy
    figures['per_class_accuracy_pt.png'] = plot_per_class_accuracy(cm)
    
    # Save all plots
    save_all_plots(figures, args.save_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults saved in: {args.save_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Visualizations: {args.save_dir}/*.png")


if __name__ == "__main__":
    main()
