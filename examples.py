"""
Example scripts demonstrating how to use the Epilepsy Detection system
"""

# ============================================================================
# EXAMPLE 1: Simple Model Training and Evaluation
# ============================================================================

def example_basic_training():
    """Basic training and evaluation pipeline"""
    import torch
    from data_loader import load_experiment_data, create_data_loaders
    from model import create_model
    from train import train_model, test_model, set_seed
    from config import TRAINING_CONFIG, MODEL_CONFIG, DATA_DIR
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load data
    print("Loading data...")
    segments, labels, file_indices, _ = load_experiment_data(DATA_DIR, 'Z', 'N')
    
    # Create dataloaders (80-20 split)
    split_idx = int(0.8 * len(segments))
    train_loader, test_loader = create_data_loaders(
        segments[:split_idx], labels[:split_idx], file_indices[:split_idx],
        segments[split_idx:], labels[split_idx:], file_indices[split_idx:],
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Create model
    print("Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(**MODEL_CONFIG)
    
    # Train
    print("Training model...")
    history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        device=device,
        save_path='results/model.pth',
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # Evaluate
    print("Evaluating model...")
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = test_model(model, test_loader, device=device, model_path='results/model.pth')
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")


# ============================================================================
# EXAMPLE 2: Using Trained Model for Predictions
# ============================================================================

def example_single_prediction():
    """Make prediction on single EEG signal"""
    import numpy as np
    from inference_utils import load_trained_model, get_prediction_confidence
    
    # Load trained model
    model = load_trained_model('results/model.pth', device='cuda')
    
    # Create dummy EEG signal (100 data points)
    eeg_signal = np.random.randn(100)
    
    # Get prediction with confidence
    pred_class, confidence, probabilities = get_prediction_confidence(model, eeg_signal)
    
    # Display results
    prediction = "🔴 SEIZURE DETECTED" if pred_class == 1 else "🟢 HEALTHY"
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Healthy Probability: {probabilities[0]:.4f}")
    print(f"Seizure Probability: {probabilities[1]:.4f}")


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Signals
# ============================================================================

def example_batch_prediction():
    """Process multiple EEG signals at once"""
    import numpy as np
    from inference_utils import load_trained_model, batch_predict
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    # Load model
    model = load_trained_model('results/model.pth')
    
    # Create batch of signals (32 samples)
    batch_signals = np.random.randn(32, 100)
    batch_labels = np.random.randint(0, 2, 32)
    
    # Convert to tensors
    signals_tensor = torch.FloatTensor(batch_signals).unsqueeze(1)
    labels_tensor = torch.LongTensor(batch_labels)
    
    # Create dataloader
    dataset = TensorDataset(signals_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Batch predict
    predictions, probabilities = batch_predict(model, dataloader)
    
    print(f"Processed {len(predictions)} samples")
    print(f"Seizures detected: {(predictions == 1).sum()}")
    print(f"Healthy classified: {(predictions == 0).sum()}")
    print(f"Mean confidence: {np.mean(probabilities):.4f}")


# ============================================================================
# EXAMPLE 4: Complete Evaluation with Metrics
# ============================================================================

def example_complete_evaluation():
    """Full evaluation with all metrics"""
    import numpy as np
    import torch
    from data_loader import load_experiment_data, create_data_loaders
    from inference_utils import load_trained_model
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, f1_score,
        accuracy_score, roc_auc_score
    )
    from config import DATA_DIR
    
    # Load data
    segments, labels, file_indices, _ = load_experiment_data(DATA_DIR, 'Z', 'N')
    split_idx = int(0.8 * len(segments))
    _, test_loader = create_data_loaders(
        segments[:split_idx], labels[:split_idx], file_indices[:split_idx],
        segments[split_idx:], labels[split_idx:], file_indices[split_idx:],
        batch_size=32
    )
    
    # Load model
    model = load_trained_model('results/model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Get predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Print results
    print("=" * 60)
    print("COMPLETE EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {auc:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"TP: {tp:4d}  FN: {fn:4d}")
    print(f"FP: {fp:4d}  TN: {tn:4d}")


# ============================================================================
# EXAMPLE 5: Data Preprocessing
# ============================================================================

def example_preprocessing():
    """How to preprocess EEG data"""
    import numpy as np
    from preprocessing import (
        resample_to_100hz, apply_filters, segment_signal, minmax_scale
    )
    
    # Simulate raw EEG data at 173.61 Hz
    original_fs = 173.61
    raw_eeg = np.random.randn(4096)
    
    # 1. Resample to 100 Hz
    resampled = resample_to_100hz(raw_eeg, original_fs, 100)
    print(f"After resampling: {len(resampled)} samples")
    
    # 2. Apply filtering
    filtered = apply_filters(resampled, fs=100)
    print(f"After filtering: {len(filtered)} samples")
    
    # 3. Segment the signal
    segments = segment_signal(filtered, segment_length=100, overlap=0)
    print(f"Created {len(segments)} segments of 100 data points each")
    
    # 4. Scale each segment
    scaled_segments = np.array([minmax_scale(seg) for seg in segments])
    print(f"Segments shape: {scaled_segments.shape}")
    print(f"Range: [{scaled_segments.min():.4f}, {scaled_segments.max():.4f}]")


# ============================================================================
# EXAMPLE 6: Using REST API
# ============================================================================

def example_rest_api():
    """How to use the REST API"""
    import requests
    import json
    
    BASE_URL = "http://localhost:8000"
    
    # Single prediction
    eeg_signal = [0.1, 0.2, 0.15, -0.1, 0.05] * 20  # 100 values
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"signal": eeg_signal}
    )
    
    result = response.json()
    print("Single Prediction Result:")
    print(json.dumps(result, indent=2))
    
    # Batch predictions
    batch_signals = [{"signal": eeg_signal} for _ in range(10)]
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_signals
    )
    
    result = response.json()
    print("\nBatch Prediction Summary:")
    print(json.dumps(result['summary'], indent=2))
    
    # Model info
    response = requests.get(f"{BASE_URL}/model/info")
    print("\nModel Info:")
    print(json.dumps(response.json(), indent=2))


# ============================================================================
# EXAMPLE 7: Streamlit Dashboard (see app.py)
# ============================================================================

def example_streamlit():
    """To run the Streamlit dashboard:
    
    streamlit run app.py
    
    This will launch an interactive dashboard with:
    - Model overview
    - Model evaluation with all metrics
    - Real-time predictions
    - Performance analysis
    
    Open http://localhost:8501 in your browser
    """
    pass


# ============================================================================
# EXAMPLE 8: Custom Training Loop
# ============================================================================

def example_custom_training():
    """Custom training with specific hyperparameters"""
    import torch
    from data_loader import load_experiment_data, create_data_loaders
    from model import create_model
    from train import train_epoch, evaluate, set_seed
    import torch.nn as nn
    import torch.optim as optim
    from config import DATA_DIR
    
    # Configuration
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    segments, labels, file_indices, _ = load_experiment_data(DATA_DIR, 'Z', 'N')
    split_idx = int(0.8 * len(segments))
    train_loader, val_loader = create_data_loaders(
        segments[:split_idx], labels[:split_idx], file_indices[:split_idx],
        segments[split_idx:], labels[split_idx:], file_indices[split_idx:],
        batch_size=32
    )
    
    # Create model
    model = create_model(
        input_channels=1,
        input_length=100,
        num_tcn_blocks=2,
        tcn_channels=128,
        kernel_size=4
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Custom training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("EPILEPSY DETECTION - USAGE EXAMPLES")
    print("=" * 70)
    print()
    print("These functions demonstrate how to use the system.")
    print("Uncomment the examples you want to run in main().")
    print()
    print("Available examples:")
    print("  1. example_basic_training() - Train and evaluate model")
    print("  2. example_single_prediction() - Single EEG prediction")
    print("  3. example_batch_prediction() - Batch predictions")
    print("  4. example_complete_evaluation() - Full metrics evaluation")
    print("  5. example_preprocessing() - Data preprocessing pipeline")
    print("  6. example_rest_api() - REST API usage")
    print("  7. example_streamlit() - Streamlit dashboard")
    print("  8. example_custom_training() - Custom training loop")
