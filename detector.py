import os
import torch
from PIL import Image
import numpy as np
from model import UrticariaDetectionModel
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def organize_images(image_source):
    images = []

    for image in os.listdir(image_source):
        image_path = os.path.join(image_source, image)

        # Load image and convert to tensor
        image = Image.open(image_path)

        # Convert PIL image to numpy array, normalize to [0,1], and convert to tensor
        image_tensor = torch.FloatTensor(np.array(image)) / 255.0

        # Add channel dimension if image is grayscale
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        images.append(image_tensor)
    
    # Stack all tensors along the batch dimension
    return torch.cat(images, dim=0) #this basically returns a list of tensors

def train_model(train_loader, val_loader, batch_size=10, epochs=100, learning_rate=0.001):
    # Initialize model
    model = UrticariaDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
    
    return model, train_losses, val_losses, train_accs, val_accs

def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store predictions and ground truth for visualization
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            # Store predictions and targets for visualization
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate final metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    print('\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return test_loss, test_acc

if __name__ == "__main__":
    image_source = r"C:\Users\evely\VisualStudioCode_Files\UrticariaDetectionModel\train"
    tensors = []

    tensors = organize_images(image_source=image_source)

    train_tensors = tensors[:116]
    val_tensors = tensors[116:136]
    test_tensors = tensors[136:]

    # Create data loaders
    train_dataset = TensorDataset(train_tensors) #i'm guessing TensorDataset() turns lists of tensors into a dataset format
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True) #this uses the dataset format created by TensorDataset() to create a DataLoader object?
    val_dataset = TensorDataset(val_tensors)
    val_loader = DataLoader(val_dataset, batch_size=10)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=10)

    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader,
        batch_size=10,
        epochs=100,
        learning_rate=0.001
    )

    # Save the model
    torch.save(model.state_dict(), 'urticaria_detection_model.pth')
    
    # Test the model
    test_loss, test_acc = test_model(model, test_loader) 

    #save model after test
    torch.save(model.state_dict(), "final_urticariaDetectionModel.pth")
