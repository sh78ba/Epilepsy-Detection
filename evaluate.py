"""
Comprehensive model evaluation script
Generates detailed metrics, visualizations, and reports
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score, classification_report
)
import os
from pathlib import Path
from data_loader import load_experiment_data, create_data_loaders
from train import evaluate
from inference_utils import load_trained_model, batch_predict


def evaluate_model_comprehensive(model_path, data_dir='data', output_dir='evaluation_results'):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing EEG data
        output_dir: Directory to save evaluation results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_trained_model(model_path, device)
    
    # Load data
    print("Loading data...")
    try:
        segments, labels, file_indices, _ = load_experiment_data(data_dir, 'Z', 'N')
        print(f"Loaded {len(segments)} segments")
        
        # Create data loaders (simple 80-20 split)
        split_idx = int(0.8 * len(segments))
        train_loader, test_loader = create_data_loaders(
            segments[:split_idx], labels[:split_idx], file_indices[:split_idx],
            segments[split_idx:], labels[split_idx:], file_indices[split_idx:],
            batch_size=32
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure data is in 'data/A', 'data/B', and 'data/E' directories")
        return
    
    # Evaluate
    print("Evaluating model...")
    criterion = torch.nn.CrossEntropyLoss()
    
    # Get predictions
    all_preds_train = []
    all_labels_train = []
    all_probs_train = []
    
    all_preds_test = []
    all_labels_test = []
    all_probs_test = []
    
    model.eval()
    with torch.no_grad():
        # Train set
        for data, target in train_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(target.cpu().numpy())
            all_probs_train.extend(probs[:, 1].cpu().numpy())
        
        # Test set
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(target.cpu().numpy())
            all_probs_test.extend(probs[:, 1].cpu().numpy())
    
    # Convert to numpy arrays
    all_preds_train = np.array(all_preds_train)
    all_labels_train = np.array(all_labels_train)
    all_probs_train = np.array(all_probs_train)
    
    all_preds_test = np.array(all_preds_test)
    all_labels_test = np.array(all_labels_test)
    all_probs_test = np.array(all_probs_test)
    
    # Calculate metrics
    def calculate_detailed_metrics(y_true, y_pred, y_proba):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'auc': auc_score,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        return metrics, cm
    
    train_metrics, train_cm = calculate_detailed_metrics(all_labels_train, all_preds_train, all_probs_train)
    test_metrics, test_cm = calculate_detailed_metrics(all_labels_test, all_preds_test, all_probs_test)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING SET METRICS")
    print("="*60)
    print(f"Accuracy:   {train_metrics['accuracy']:.4f}")
    print(f"Precision:  {train_metrics['precision']:.4f}")
    print(f"Recall:     {train_metrics['recall']:.4f}")
    print(f"F1-Score:   {train_metrics['f1']:.4f}")
    print(f"Specificity: {train_metrics['specificity']:.4f}")
    print(f"ROC-AUC:    {train_metrics['auc']:.4f}")
    print(f"TP: {train_metrics['tp']}, TN: {train_metrics['tn']}, FP: {train_metrics['fp']}, FN: {train_metrics['fn']}")
    
    print("\n" + "="*60)
    print("TEST SET METRICS")
    print("="*60)
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"F1-Score:   {test_metrics['f1']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"ROC-AUC:    {test_metrics['auc']:.4f}")
    print(f"TP: {test_metrics['tp']}, TN: {test_metrics['tn']}, FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("="*60)
    print(classification_report(all_labels_test, all_preds_test, 
                               target_names=['Healthy', 'Seizure']))
    
    # Save metrics to JSON
    metrics_for_json = {
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in ['fpr', 'tpr']},
        'test_metrics': {k: v for k, v in test_metrics.items() if k not in ['fpr', 'tpr']},
        'train_confusion_matrix': train_cm.tolist(),
        'test_confusion_matrix': test_cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_for_json, f, indent=2)
    
    print(f"\n✓ Metrics saved to {os.path.join(output_dir, 'metrics.json')}")
    
    # Create visualizations
    create_evaluation_visualizations(
        train_cm, test_cm,
        train_metrics, test_metrics,
        all_labels_test, all_probs_test,
        output_dir
    )
    
    return train_metrics, test_metrics


def create_evaluation_visualizations(train_cm, test_cm, train_metrics, test_metrics, 
                                     y_test, y_proba_test, output_dir='evaluation_results'):
    """
    Create comprehensive visualizations
    """
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Confusion Matrices
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Healthy', 'Seizure'],
                yticklabels=['Healthy', 'Seizure'])
    ax1.set_title('Training Set Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Healthy', 'Seizure'],
                yticklabels=['Healthy', 'Seizure'])
    ax2.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 2. Metrics Comparison
    ax3 = plt.subplot(2, 3, 3)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    train_vals = [train_metrics['accuracy'], train_metrics['precision'], 
                  train_metrics['recall'], train_metrics['f1'], train_metrics['specificity']]
    test_vals = [test_metrics['accuracy'], test_metrics['precision'],
                test_metrics['recall'], test_metrics['f1'], test_metrics['specificity']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax3.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    ax3.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    # 3. ROC Curves
    ax4 = plt.subplot(2, 3, 4)
    fpr_test = np.array(test_metrics['fpr'])
    tpr_test = np.array(test_metrics['tpr'])
    ax4.plot(fpr_test, tpr_test, color='darkorange', lw=2, 
            label=f"ROC Curve (AUC = {test_metrics['auc']:.4f})")
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve (Test Set)', fontsize=12, fontweight='bold')
    ax4.legend(loc="lower right")
    ax4.grid(alpha=0.3)
    
    # 4. Prediction Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_proba_test[y_test == 0], bins=30, alpha=0.7, label='Healthy (True)', color='green')
    ax5.hist(y_proba_test[y_test == 1], bins=30, alpha=0.7, label='Seizure (True)', color='red')
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax5.set_xlabel('Seizure Probability')
    ax5.set_ylabel('Count')
    ax5.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 5. Key Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    TEST SET PERFORMANCE SUMMARY
    
    Accuracy:            {test_metrics['accuracy']:.4f}
    Precision (PPV):     {test_metrics['precision']:.4f}
    Recall (Sensitivity):{test_metrics['recall']:.4f}
    Specificity (TNR):   {test_metrics['specificity']:.4f}
    F1-Score:            {test_metrics['f1']:.4f}
    ROC-AUC:             {test_metrics['auc']:.4f}
    
    True Positives:      {test_metrics['tp']}
    True Negatives:      {test_metrics['tn']}
    False Positives:     {test_metrics['fp']}
    False Negatives:     {test_metrics['fn']}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'evaluation_report.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Find latest model
        from inference_utils import find_latest_model
        model_path = find_latest_model()
        
        if model_path is None:
            print("Error: No trained model found. Please train a model first.")
            sys.exit(1)
    
    print(f"Using model: {model_path}")
    evaluate_model_comprehensive(model_path)
