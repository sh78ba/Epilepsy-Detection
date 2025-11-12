import os
import argparse
import json
import torch
import numpy as np
from data_loader import load_experiment_data, create_participant_level_splits, create_data_loaders
from model import create_model
from train import (
    set_seed, train_model, test_model, print_metrics, 
    calculate_mean_std, print_cross_validation_results
)


def run_single_fold(experiment_type, fold_idx, segments, labels, file_indices, 
                   num_epochs=100, device='cuda', results_dir='results'):
    """
    Run a single fold of cross-validation.
    """
    print("\n" + "=" * 80)
    print(f"Running Fold {fold_idx + 1} - {experiment_type} - TCN_SA")
    print("=" * 80)
    
    # Create participant-level splits
    splits = list(create_participant_level_splits(segments, labels, file_indices, 
                                                  n_splits=3, random_state=42))
    train_segments, train_labels, test_segments, test_labels, test_file_idx = splits[fold_idx]
    
    # Create dummy file indices for data loaders
    train_file_indices = np.arange(len(train_segments))
    test_file_indices = np.arange(len(test_segments))
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_segments, train_labels, train_file_indices,
        test_segments, test_labels, test_file_indices,
        batch_size=12
    )
    
    # Create model (TCN-SA only)
    model = create_model(input_channels=1, input_length=100, 
                       num_tcn_blocks=2, tcn_channels=128, kernel_size=4)
    
    # Setup save path
    save_dir = os.path.join(results_dir, experiment_type, 'TCN_SA')
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f'model_fold{fold_idx+1}.pth')
    
    # Train
    history = train_model(
        model, train_loader, test_loader,
        num_epochs=num_epochs,
        learning_rate=0.01,
        device=device,
        save_path=model_save_path,
        early_stopping_patience=20
    )
    
    # Test with best model
    test_metrics = test_model(model, test_loader, device=device, model_path=model_save_path)
    
    print("\n" + "-" * 80)
    print(f"Fold {fold_idx + 1} Test Results:")
    print_metrics(test_metrics, prefix="  ")
    print("-" * 80)
    
    # Save fold results
    fold_results = {
        'fold': fold_idx + 1,
        'test_metrics': {k: float(v) if v is not None else None for k, v in test_metrics.items()},
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    results_file = os.path.join(save_dir, f'fold{fold_idx+1}_results.json')
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    return test_metrics


def run_experiment(data_dir, experiment_type, num_epochs=100, 
                  device='cuda', results_dir='results'):
    """
    Run a complete experiment with 3-fold cross-validation.
    """
    print("\n" + "=" * 80)
    print(f"Starting Experiment: {experiment_type} with TCN_SA")
    print("=" * 80)
    
    # Parse experiment type
    if experiment_type == 'A_vs_E':
        set1, set2 = 'A', 'E'
    elif experiment_type == 'B_vs_E':
        set1, set2 = 'B', 'E'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Load data
    print("\nLoading and preprocessing data...")
    segments, labels, file_indices, file_to_set_map = load_experiment_data(
        data_dir, set1, set2
    )
    
    # Run 3-fold cross-validation
    all_metrics = []
    
    for fold_idx in range(3):
        fold_metrics = run_single_fold(
            experiment_type, fold_idx, segments, labels, file_indices,
            num_epochs=num_epochs, device=device, results_dir=results_dir
        )
        all_metrics.append(fold_metrics)
    
    # Calculate mean and SE across folds
    cv_results = calculate_mean_std(all_metrics)
    
    # Print final results
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {experiment_type} - TCN_SA")
    print_cross_validation_results(cv_results)
    
    # Save final results
    save_dir = os.path.join(results_dir, experiment_type, 'TCN_SA')
    final_results = {
        'experiment': experiment_type,
        'model': 'TCN_SA',
        'n_folds': 3,
        'cv_results': {
            metric: {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'se': float(stats['se']),
                'values': [float(v) for v in stats['values']]
            }
            for metric, stats in cv_results.items()
        }
    }
    
    final_results_file = os.path.join(save_dir, 'final_results.json')
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    
    return cv_results


def run_all_experiments(data_dir, num_epochs=100, device='cuda', results_dir='results'):
    """
    Run both A vs E and B vs E experiments with TCN_SA model.
    """
    experiments = ['A_vs_E', 'B_vs_E']
    
    all_results = {}
    
    for exp_type in experiments:
        results = run_experiment(
            data_dir, exp_type, num_epochs=num_epochs, 
            device=device, results_dir=results_dir
        )
        all_results[exp_type] = results
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL EXPERIMENTS")
    print("=" * 80)
    
    for exp_type, results in all_results.items():
        print(f"\n{exp_type}:")
        for metric, stats in results.items():
            if metric == 'accuracy':
                print(f"  {metric.capitalize()}: {stats['mean']*100:.2f}% ± {stats['se']*100:.2f}%")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Run Bonn EEG Epilepsy Detection with TCN-SA')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing A, B, E folders')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'A_vs_E', 'B_vs_E'],
                       help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("=" * 80)
    print("Epilepsy Detection Using TCN-SA")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Model: TCN-SA")
    print("=" * 80)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please ensure you have downloaded the Bonn dataset.")
        return
    
    # Run experiments
    if args.experiment == 'all':
        run_all_experiments(args.data_dir, args.epochs, args.device, args.results_dir)
    else:
        run_experiment(args.data_dir, args.experiment, 
                      args.epochs, args.device, args.results_dir)


if __name__ == '__main__':
    main()
