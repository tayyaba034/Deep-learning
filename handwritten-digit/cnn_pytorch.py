"""
CNN Model Implementation using PyTorch
Handwritten Digit Classification on MNIST Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class SimpleCNN(nn.Module):
    """Basic CNN architecture for MNIST digit classification"""
    
    def __init__(self, num_classes=10):
        """
        Initialize the Simple CNN model
        
        Args:
            num_classes: Number of output classes (10 for MNIST)
        """
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block: Conv -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second block: Conv -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ImprovedCNN(nn.Module):
    """Advanced CNN architecture with batch normalization"""
    
    def __init__(self, num_classes=10):
        """
        Initialize the Improved CNN model
        
        Args:
            num_classes: Number of output classes
        """
        super(ImprovedCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelTrainer:
    """Utility class for training PyTorch models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model to train
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
    def setup_training(self, learning_rate=0.001, optimizer_type='adam'):
        """
        Setup optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        """
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("Simple CNN Architecture")
    print("=" * 60)
    simple_model = SimpleCNN()
    print(simple_model)
    print(f"\nTotal Parameters: {simple_model.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Improved CNN Architecture")
    print("=" * 60)
    improved_model = ImprovedCNN()
    print(improved_model)
    print(f"\nTotal Parameters: {improved_model.count_parameters():,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    simple_output = simple_model(dummy_input)
    improved_output = improved_model(dummy_input)
    
    print(f"\nSimple CNN output shape: {simple_output.shape}")
    print(f"Improved CNN output shape: {improved_output.shape}")
