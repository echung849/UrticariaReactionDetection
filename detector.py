import os
import torch
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import UrticariaDetectionModel

def organize_images(image_source, target_size=(224, 224)):
    """
    Load and organize images from a directory into tensors with consistent dimensions.
    
    Args:
        image_source (str): Path to directory containing images
        target_size (tuple): Target size for all images (width, height)
        
    Returns:
        torch.Tensor: Tensor containing all images
        list: List of image filenames
    """
    images = []
    filenames = []

    if not os.path.exists(image_source):
        raise FileNotFoundError(f"Image directory not found: {image_source}")
        
    image_files = [f for f in os.listdir(image_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {image_source}")
    
    print(f"Found {len(image_files)} images in {image_source}")

    for image in image_files:
        image_path = os.path.join(image_source, image)
        
        try:
            # Load image and resize to target size
            img = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels
            img = img.resize(target_size)
            
            # Convert PIL image to numpy array, normalize to [0,1], and convert to tensor
            # Shape: (height, width, channels)
            img_array = np.array(img) / 255.0
            
            # Convert to PyTorch tensor and rearrange to (channels, height, width) - PyTorch convention
            image_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            images.append(image_tensor)
            filenames.append(image)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
    
    if not images:
        raise ValueError("No images were successfully loaded")
        
    # Stack all tensors along the batch dimension
    return torch.cat(images, dim=0), filenames

def prepare_data(images, labels=None, train_ratio=0.7, val_ratio=0.15, batch_size=10):
    """
    Prepare data for training, validation, and testing.
    
    Args:
        images (torch.Tensor): Tensor containing all images
        labels (torch.Tensor, optional): Tensor containing labels (if available)
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: Train, validation and test data loaders
    """
    num_samples = images.size(0)
    
    if labels is None:
        # If no labels provided, create dummy labels (for development purposes)
        print("WARNING: Using randomly generated labels for demonstration purposes.")
        print("In a real application, you should provide actual labels.")
        labels = torch.randint(0, 2, (num_samples,))
    
    # Calculate split sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    
    print(f"Data split: {train_size} training, {val_size} validation, {test_size} test samples")
    
    # Create full dataset
    dataset = TensorDataset(images, labels)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, device='cpu', batch_size=10, epochs=100, learning_rate=0.001):
    """
    Train the urticaria detection model.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (str): Device to train on ('cpu' or 'cuda')
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tuple: Trained model and training metrics
    """
    # Check if GPU is available
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Training on device: {device}")
    
    # Get a sample batch to determine input size
    for batch_X, _ in train_loader:
        # Get input shape from first batch
        input_shape = batch_X.shape
        print(f"Input batch shape: {input_shape}")
        
        # Calculate flattened input size for the model
        input_size = np.prod(input_shape[1:])
        print(f"Flattened input size: {input_size}")
        break
        
    # Initialize model with correct input size
    model = UrticariaDetectionModel(input_size=input_size)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            # Reshape input to 2D before passing to model
            batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten all dimensions except batch
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Reshape input to 2D before passing to model
                batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten all dimensions except batch
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
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
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            torch.save(model.state_dict(), 'best_urticaria_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_urticaria_model.pth'))
    
    # Return the model and training history
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def test_model(model, test_loader, device='cpu'):
    """
    Test the trained model on test data.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to test on ('cpu' or 'cuda')
        
    Returns:
        tuple: Test metrics
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store predictions and ground truth for visualization
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # Reshape input to 2D before passing to model
            batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten all dimensions except batch
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            # Store predictions and targets for visualization
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate final metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    print('\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Classification report and confusion matrix
    try:
        print('\nClassification Report:')
        print(classification_report(all_targets, all_predictions, target_names=['Non-Urticaria', 'Urticaria']))
        
        cm = confusion_matrix(all_targets, all_predictions)
        print('\nConfusion Matrix:')
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Non-Urticaria', 'Urticaria']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        print('Confusion matrix saved as confusion_matrix.png')
    except Exception as e:
        print(f"Error generating classification metrics: {e}")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }

def save_training_plots(history):
    """
    Generate and save training plots.
    
    Args:
        history (dict): Training history dictionary
    """
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Training Accuracy')
    plt.plot(history['val_accs'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print('Training history plots saved as training_history.png')

if __name__ == "__main__":
    # Path to your image data
    image_source = "train"  # Update this path to match your directory structure
    
    if not os.path.exists(image_source):
        print(f"WARNING: Directory {image_source} not found. Creating empty directory.")
        os.makedirs(image_source, exist_ok=True)
        print(f"Please add image files to the {image_source} directory before running.")
        exit(1)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load and organize images with consistent size
        print("Loading and organizing images...")
        images, filenames = organize_images(image_source)
        
        print(f"Loaded {len(filenames)} images.")
        print(f"Image tensor shape: {images.shape}")
        
        # For demonstration - need actual labels in real application
        # Here we're creating random labels (0 or 1) for demonstration
        # In a real scenario, you'd load actual labels from file or directory structure
        labels = torch.randint(0, 2, (images.size(0),))
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data(
            images, labels, train_ratio=0.7, val_ratio=0.15, batch_size=10
        )
        
        # Train the model
        print("Starting model training...")
        model, history = train_model(
            train_loader, val_loader,
            device=device,
            batch_size=10,
            epochs=100,
            learning_rate=0.001
        )
        
        # Save training plots
        save_training_plots(history)
        
        # Test the model
        print("\nEvaluating model on test data...")
        test_results = test_model(model, test_loader, device=device)
        
        # Save the final model
        torch.save(model.state_dict(), "final_urticaria_detection_model.pth")
        print("\nModel training and evaluation complete!")
        print("Final model saved as 'final_urticaria_detection_model.pth'")
        
    except Exception as e:
        print(f"Error in execution: {e}")
