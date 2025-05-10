import torch
import torch.nn as nn
import torch.nn.functional as F

class UrticariaDetectionModel(nn.Module):
    def __init__(self, input_size=150528, hidden_size=256, num_classes=2):
        """
        Initialize the Urticaria Detection Model.
        
        Args:
            input_size (int): Size of the flattened input features
                              Default is 150528 (224x224x3 image)
            hidden_size (int): Size of hidden layers
            num_classes (int): Number of output classes (2 for binary classification)
        """
        super(UrticariaDetectionModel, self).__init__()
        
        # Scale hidden size based on input size to prevent overfitting
        # For very large inputs, we increase the network capacity
        if input_size > 100000:
            hidden_size = 512
        
        # Print model configuration for debugging
        print(f"Initializing model with input_size={input_size}, hidden_size={hidden_size}")
        
        # Input layer with batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Dropout for regularization - higher rate for larger networks
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Input normalization
        x = self.input_bn(x)
        
        # First hidden layer with ReLU and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer with ReLU and dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third hidden layer with ReLU and dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation - will be applied in loss function)
        x = self.fc4(x)
        
        return x
