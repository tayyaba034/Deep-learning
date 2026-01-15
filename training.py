"""
Training Pipeline for Cat vs Dog Classification
Features:
- Custom learning rate schedules
- Multiple callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Training with mixed precision
- Ensemble training support
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from datetime import datetime
import json


class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate schedule with warmup"""
    
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps=0, alpha=0.0):
        """
        Initialize learning rate schedule
        
        Args:
            initial_learning_rate: Initial learning rate
            decay_steps: Number of steps to decay over
            warmup_steps: Number of warmup steps
            alpha: Minimum learning rate as fraction of initial_learning_rate
        """
        super(CosineDecayWithWarmup, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        if step < warmup_steps:
            # Linear warmup
            return self.initial_learning_rate * (step / warmup_steps)
        else:
            # Cosine decay
            step = step - warmup_steps
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * step / decay_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.initial_learning_rate * decayed


class TrainingConfig:
    """Training configuration"""
    
    def __init__(self):
        self.img_size = 224
        self.batch_size = 32
        self.epochs = 50
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.use_mixed_precision = True
        self.augmentation_level = 'medium'
        self.use_mixup_cutmix = True
        self.mixup_alpha = 0.2
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, filepath):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config=None):
        """
        Initialize training pipeline
        
        Args:
            config: TrainingConfig instance
        """
        self.config = config if config else TrainingConfig()
        
        # Enable mixed precision training
        if self.config.use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled (float16)")
        
        self.model = None
        self.history = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def compile_model(self, model):
        """
        Compile model with optimizer and loss
        
        Args:
            model: Keras model to compile
        """
        # Create learning rate schedule
        steps_per_epoch = 1000  # Will be updated during training
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        lr_schedule = CosineDecayWithWarmup(
            initial_learning_rate=self.config.initial_lr,
            decay_steps=total_steps - warmup_steps,
            warmup_steps=warmup_steps,
            alpha=0.01
        )
        
        # Create optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        print(f"Model compiled with {optimizer.__class__.__name__} optimizer")
    
    def get_callbacks(self, model_save_path):
        """
        Create training callbacks
        
        Args:
            model_save_path: Path to save best model
        
        Returns:
            List of callbacks
        """
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=f'logs/{self.timestamp}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=f'training_log_{self.timestamp}.csv',
                separator=',',
                append=False
            )
        ]
        
        return callback_list
    
    def train(self, train_generator, val_generator, steps_per_epoch, 
              validation_steps, model_save_path='best_model.keras'):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            model_save_path: Path to save the best model
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not compiled. Call compile_model() first.")
        
        # Get callbacks
        callback_list = self.get_callbacks(model_save_path)
        
        # Train model
        print("\n" + "="*80)
        print(f"Starting training at {self.timestamp}")
        print("="*80)
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        return self.history
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(epochs, history['auc'], 'b-', label='Training AUC', linewidth=2)
        axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Validation AUC', linewidth=2)
        axes[1, 0].set_title('AUC', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 1].plot(epochs, history['precision'], 'b-', label='Precision', linewidth=2)
        axes[1, 1].plot(epochs, history['recall'], 'g-', label='Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        
        return fig
    
    def evaluate(self, test_generator, steps):
        """
        Evaluate model on test set
        
        Args:
            test_generator: Test data generator
            steps: Number of steps
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        results = self.model.evaluate(test_generator, steps=steps, verbose=1)
        
        metrics_names = self.model.metrics_names
        results_dict = dict(zip(metrics_names, results))
        
        print("\n" + "="*80)
        print("Test Results:")
        print("="*80)
        for name, value in results_dict.items():
            print(f"{name.capitalize()}: {value:.4f}")
        print("="*80)
        
        return results_dict


if __name__ == "__main__":
    # Test training configuration
    print("Testing training configuration...")
    
    config = TrainingConfig()
    print("\nTraining Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Test learning rate schedule
    print("\nTesting learning rate schedule...")
    lr_schedule = CosineDecayWithWarmup(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        warmup_steps=100,
        alpha=0.01
    )
    
    steps = [0, 50, 100, 200, 500, 800, 1000]
    print("Step -> Learning Rate:")
    for step in steps:
        lr = lr_schedule(step)
        print(f"  {step:4d} -> {lr:.6f}")
