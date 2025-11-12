

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import time
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Make cuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC (requires probabilities)
    auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': recall,  # Same as recall
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    metrics['loss'] = avg_loss
    
    return metrics


def train_model(model, train_loader, val_loader, num_epochs=100, 
                learning_rate=0.01, device='cuda', save_path=None,
                early_stopping_patience=20):
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        if val_metrics['auc'] is not None:
            history['val_auc'].append(val_metrics['auc'])
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            if val_metrics['auc'] is not None:
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                    'val_loss': best_val_loss,
                }, save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history


def test_model(model, test_loader, device='cuda', model_path=None):
    
    # Load best model if path provided
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    return test_metrics


def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    print(f"{prefix}Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    if metrics['auc'] is not None:
        print(f"  ROC AUC:     {metrics['auc']:.4f}")
    print(f"  Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")


def calculate_mean_std(metrics_list):
    
    result = {}
    
    # Get all metric names
    metric_names = [k for k in metrics_list[0].keys() if k not in ['tp', 'tn', 'fp', 'fn', 'loss']]
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation
            se = std / np.sqrt(len(values))  # Standard error
            
            result[metric_name] = {
                'mean': mean,
                'std': std,
                'se': se,
                'values': values
            }
    
    return result


def print_cross_validation_results(results):
    
    print("\nCross-Validation Results (Mean ± SE):")
    print("=" * 60)
    
    for metric_name, stats in results.items():
        mean = stats['mean']
        se = stats['se']
        print(f"{metric_name.capitalize():12s}: {mean:.4f} ± {se:.4f} ({mean*100:.2f}% ± {se*100:.2f}%)")
    
    print("=" * 60)
