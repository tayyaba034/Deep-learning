"""
Custom CNN Architecture with Attention Mechanisms for Cat vs Dog Classification
Features:
- Squeeze-and-Excitation (SE) blocks for channel attention
- Spatial attention mechanism
- Residual connections for better gradient flow
- Multi-scale feature extraction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


@keras.utils.register_keras_serializable(package="Custom")
class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, filters, ratio=16, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        
    def build(self, input_shape):
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            layers.Dense(self.filters // self.ratio, activation='relu'),
            layers.Dense(self.filters, activation='sigmoid')
        ])
        
    def call(self, inputs):
        # Squeeze: Global average pooling
        squeeze = self.squeeze(inputs)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excitation = self.excitation(squeeze)
        # Reshape to (batch, 1, 1, channels)
        excitation = tf.reshape(excitation, [-1, 1, 1, self.filters])
        # Scale the input
        return inputs * excitation


@keras.utils.register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, 
                                   padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # Average and max pooling along channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention


@keras.utils.register_keras_serializable(package="Custom")
class AttentionResidualBlock(layers.Layer):
    """Residual block with attention mechanisms"""
    
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(AttentionResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
    def build(self, input_shape):
        # Main path
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, 
                                    strides=self.strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, 
                                    padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Attention mechanisms
        self.se_block = SqueezeExcitation(self.filters)
        self.spatial_attention = SpatialAttention()
        
        # Shortcut connection
        if self.strides > 1 or input_shape[-1] != self.filters:
            self.shortcut = keras.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.strides),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Apply attention
        x = self.se_block(x)
        x = self.spatial_attention(x)
        
        # Add shortcut
        shortcut = self.shortcut(inputs)
        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        
        return x


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    Build custom CNN with attention mechanisms
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks with attention
    # Stage 1: 64 filters
    x = AttentionResidualBlock(64)(x)
    x = AttentionResidualBlock(64)(x)
    
    # Stage 2: 128 filters
    x = AttentionResidualBlock(128, strides=2)(x)
    x = AttentionResidualBlock(128)(x)
    
    # Stage 3: 256 filters
    x = AttentionResidualBlock(256, strides=2)(x)
    x = AttentionResidualBlock(256)(x)
    
    # Stage 4: 512 filters
    x = AttentionResidualBlock(512, strides=2)(x)
    x = AttentionResidualBlock(512)(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AttentionCatDogCNN')
    
    return model


def build_efficientnet_transfer(input_shape=(224, 224, 3), num_classes=1):
    """
    Build transfer learning model using EfficientNetB3
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
    
    Returns:
        Keras Model
    """
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='EfficientNetCatDog')
    
    return model


def build_ensemble_model(input_shape=(224, 224, 3)):
    """
    Build an ensemble of multiple models for improved accuracy
    
    Args:
        input_shape: Input image shape
    
    Returns:
        List of Keras Models
    """
    models = []
    
    # Model 1: Custom CNN with Attention
    models.append(build_custom_cnn(input_shape))
    
    # Model 2: EfficientNet Transfer Learning
    models.append(build_efficientnet_transfer(input_shape))
    
    return models


if __name__ == "__main__":
    # Test model building
    print("Building Custom CNN with Attention...")
    model = build_custom_cnn()
    model.summary()
    
    print("\n" + "="*80)
    print("Total parameters:", model.count_params())
    
    # Test with random input
    test_input = tf.random.normal((1, 224, 224, 3))
    output = model(test_input, training=False)
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output value: {output.numpy()[0][0]:.4f}")
