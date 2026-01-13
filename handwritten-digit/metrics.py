"""
Evaluation Metrics and Utilities
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix array
    """
    # For PyTorch, convert tensors to numpy if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    return cm


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # For PyTorch
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'per_class_precision': per_class_precision * 100,
        'per_class_recall': per_class_recall * 100,
        'per_class_f1': per_class_f1 * 100
    }
    
    return metrics


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 70)


def print_metrics_summary(metrics):
    """
    Print metrics summary
    
    Args:
        metrics: Dictionary of metrics
    """
    print("=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Weighted Precision: {metrics['precision']:.2f}%")
    print(f"Weighted Recall:    {metrics['recall']:.2f}%")
    print(f"Weighted F1-Score:  {metrics['f1_score']:.2f}%")
    print("=" * 70)
    
    print("\nPer-Class Performance:")
    print("-" * 70)
    print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 70)
    
    for i in range(len(metrics['per_class_precision'])):
        print(f"{i:<10} {metrics['per_class_precision'][i]:>10.2f}%    "
              f"{metrics['per_class_recall'][i]:>10.2f}%    "
              f"{metrics['per_class_f1'][i]:>10.2f}%")
    print("=" * 70)


def analyze_errors(y_true, y_pred):
    """
    Analyze prediction errors
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with error analysis
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    # Find errors
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    # Most confused pairs
    error_pairs = {}
    for idx in error_indices:
        pair = (int(y_true[idx]), int(y_pred[idx]))
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
    
    analysis = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_true) * 100,
        'error_indices': error_indices,
        'most_confused_pairs': sorted_pairs[:10]
    }
    
    return analysis


def print_error_analysis(analysis):
    """
    Print error analysis
    
    Args:
        analysis: Error analysis dictionary
    """
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print(f"Total Errors: {analysis['total_errors']}")
    print(f"Error Rate: {analysis['error_rate']:.2f}%")
    print("\nMost Confused Digit Pairs:")
    print("-" * 70)
    print(f"{'True Label':<15} {'Predicted As':<15} {'Count':<15}")
    print("-" * 70)
    
    for (true_label, pred_label), count in analysis['most_confused_pairs']:
        print(f"{true_label:<15} {pred_label:<15} {count:<15}")
    
    print("=" * 70)


def calculate_top_k_accuracy(y_true, y_pred_probs, k=3):
    """
    Calculate top-k accuracy
    
    Args:
        y_true: True labels
        y_pred_probs: Prediction probabilities
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred_probs, 'numpy'):
        y_pred_probs = y_pred_probs.numpy()
    
    # Get top-k predictions
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    top_k_acc = correct / len(y_true) * 100
    return top_k_acc


def model_summary_statistics(model):
    """
    Print model summary statistics
    
    Args:
        model: Trained model
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    
    print("=" * 70)


def save_metrics_to_file(metrics, filepath='results/metrics.txt'):
    """
    Save metrics to a text file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("METRICS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Overall Accuracy:   {metrics['accuracy']:.2f}%\n")
        f.write(f"Weighted Precision: {metrics['precision']:.2f}%\n")
        f.write(f"Weighted Recall:    {metrics['recall']:.2f}%\n")
        f.write(f"Weighted F1-Score:  {metrics['f1_score']:.2f}%\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
        f.write("-" * 70 + "\n")
        
        for i in range(len(metrics['per_class_precision'])):
            f.write(f"{i:<10} {metrics['per_class_precision'][i]:>10.2f}%    "
                   f"{metrics['per_class_recall'][i]:>10.2f}%    "
                   f"{metrics['per_class_f1'][i]:>10.2f}%\n")
        f.write("=" * 70 + "\n")
    
    print(f"Metrics saved to: {filepath}")


if __name__ == "__main__":
    # Test metrics functions with dummy data
    print("Testing metrics functions...")
    
    # Create dummy predictions
    y_true = np.random.randint(0, 10, 100)
    y_pred = np.random.randint(0, 10, 100)
    
    # Test metrics calculation
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics_summary(metrics)
    
    # Test error analysis
    analysis = analyze_errors(y_true, y_pred)
    print_error_analysis(analysis)
    
    print("\nMetrics tests completed!")
