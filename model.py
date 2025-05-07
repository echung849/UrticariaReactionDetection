import torch
import torch.nn as nn
import torch.nn.functional as F

class UrticariaDetectionModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_classes=2):
        super(UrticariaDetectionModel, self).__init__()
        
        # Input layer with batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
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
        
        # Output layer
        x = self.fc4(x)
        
        return x 