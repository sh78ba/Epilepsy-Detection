"""
Inference utilities for Epilepsy Detection model
Handles model loading, prediction, and inference pipeline
"""

import torch
import numpy as np
from model import create_model
import os


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model in evaluation mode
    """
    # Create model architecture
    model = create_model(
        input_channels=1,
        input_length=100,
        num_tcn_blocks=2,
        tcn_channels=128,
        kernel_size=4
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Handle old format
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_on_data(model, data, device='cpu', return_probabilities=False):
    """
    Make predictions on input data
    
    Args:
        model: Trained model
        data: Input data (batch or single sample)
        device: Device to use
        return_probabilities: If True, return probabilities; if False, return class predictions
    
    Returns:
        predictions: Class predictions or probabilities
    """
    # Ensure data is torch tensor
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    
    # Add channel dimension if needed
    if data.dim() == 2:
        data = data.unsqueeze(1)
    
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        
        if return_probabilities:
            return probabilities.cpu().numpy()
        else:
            predictions = torch.argmax(probabilities, dim=1)
            return predictions.cpu().numpy()


def get_model_summary(model):
    """
    Get a summary of the model architecture
    
    Args:
        model: PyTorch model
    
    Returns:
        summary_str: String representation of model architecture
    """
    summary_str = str(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
{summary_str}

{'='*60}
Model Summary:
{'='*60}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}
{'='*60}
Model Type: TCN with Self-Attention
Input Shape: (batch_size, 1, 100)
Output Shape: (batch_size, 2)
Classes: [Healthy, Seizure]
"""
    
    return summary


def batch_predict(model, data_loader, device='cpu'):
    """
    Make predictions on a full data loader
    
    Args:
        model: Trained model
        data_loader: DataLoader with input data
        device: Device to use
    
    Returns:
        all_predictions: All predictions
        all_probabilities: All probability scores
    """
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def get_prediction_confidence(model, data, device='cpu'):
    """
    Get prediction and confidence score
    
    Args:
        model: Trained model
        data: Input sample
        device: Device to use
    
    Returns:
        prediction: Class prediction (0 or 1)
        confidence: Confidence score (0-1)
        probabilities: Full probability distribution
    """
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    
    if data.dim() == 1:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        data = data.unsqueeze(1)
    
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = max(probabilities).item()
        
        return prediction, confidence, probabilities.cpu().numpy()


def find_latest_model(results_dir='results'):
    """
    Find the latest trained model in results directory
    
    Args:
        results_dir: Directory containing model results
    
    Returns:
        model_path: Path to latest model or None
    """
    latest_model = None
    latest_time = 0
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.pth'):
                filepath = os.path.join(root, file)
                file_time = os.path.getmtime(filepath)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_model = filepath
    
    return latest_model


def evaluate_on_batch(model, data, labels, device='cpu'):
    """
    Evaluate model on a batch of data
    
    Args:
        model: Trained model
        data: Input data batch
        labels: Ground truth labels
        device: Device to use
    
    Returns:
        predictions: Model predictions
        probabilities: Prediction probabilities
        accuracy: Batch accuracy
    """
    from sklearn.metrics import accuracy_score
    
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    if isinstance(labels, np.ndarray):
        labels = torch.LongTensor(labels)
    
    data = data.to(device)
    labels = labels.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        accuracy = accuracy_score(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        )
        
        return (
            predictions.cpu().numpy(),
            probabilities.cpu().numpy(),
            accuracy
        )
