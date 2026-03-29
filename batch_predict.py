"""
Batch prediction script for processing multiple EEG signals
Useful for clinical batch processing and research studies
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import os
from inference_utils import load_trained_model, get_prediction_confidence
from data_loader import load_experiment_data, create_data_loaders
from preprocessing import preprocess_eeg_file


def batch_predict_from_files(file_paths, model, device='cpu', output_file=None):
    """
    Predict on multiple EEG files
    
    Args:
        file_paths: List of paths to EEG files
        model: Trained model
        device: Device to use
        output_file: Optional output CSV file
    
    Returns:
        results: DataFrame with predictions and confidences
    """
    
    results = []
    
    for idx, file_path in enumerate(file_paths):
        try:
            # Preprocess
            segments = preprocess_eeg_file(file_path)
            
            # Predict on mean of segments
            segment_predictions = []
            for segment in segments:
                pred, conf, probs = get_prediction_confidence(model, segment, device)
                segment_predictions.append({
                    'prediction': pred,
                    'confidence': conf,
                    'healthy_prob': probs[0],
                    'seizure_prob': probs[1]
                })
            
            # Aggregate results
            predictions = [p['prediction'] for p in segment_predictions]
            confidences = [p['confidence'] for p in segment_predictions]
            mean_seizure_prob = np.mean([p['seizure_prob'] for p in segment_predictions])
            
            final_pred = 1 if mean_seizure_prob > 0.5 else 0
            
            results.append({
                'file': Path(file_path).name,
                'prediction': 'Seizure' if final_pred == 1 else 'Healthy',
                'confidence': np.mean(confidences),
                'mean_seizure_probability': mean_seizure_prob,
                'num_segments': len(segments),
                'segment_agreement': np.sum(np.array(predictions) == final_pred) / len(predictions)
            })
            
            print(f"[{idx+1}/{len(file_paths)}] {Path(file_path).name}: {results[-1]['prediction']} (conf: {results[-1]['confidence']:.4f})")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append({
                'file': Path(file_path).name,
                'prediction': 'ERROR',
                'confidence': 0,
                'mean_seizure_probability': 0,
                'num_segments': 0,
                'segment_agreement': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
    
    return df


def batch_predict_from_data(segments, model, device='cpu'):
    """
    Predict on preprocessed segments
    
    Args:
        segments: Array of shape (N, 100)
        model: Trained model
        device: Device to use
    
    Returns:
        predictions: Array of predictions
        confidences: Array of confidence scores
    """
    
    predictions = []
    confidences = []
    
    model.eval()
    with torch.no_grad():
        for segment in segments:
            pred, conf, _ = get_prediction_confidence(model, segment, device)
            predictions.append(pred)
            confidences.append(conf)
    
    return np.array(predictions), np.array(confidences)


def main():
    parser = argparse.ArgumentParser(
        description='Batch prediction for EEG seizure detection'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Directory containing EEG files or CSV with file paths')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--format', type=str, default='txt',
                       help='EEG file format (txt or csv)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_trained_model(args.model, args.device)
    
    # Get file paths
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Get all files from directory
        file_paths = list(input_path.glob(f'*.{args.format}'))
    elif input_path.suffix == '.csv':
        # Read file paths from CSV
        df = pd.read_csv(args.input)
        file_paths = df[df.columns[0]].tolist()
        file_paths = [Path(f) for f in file_paths if Path(f).exists()]
    else:
        file_paths = [input_path]
    
    print(f"Found {len(file_paths)} files to process")
    
    # Batch predict
    results = batch_predict_from_files(file_paths, model, args.device, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(results.to_string(index=False))
    print(f"\nSeizure cases: {(results['prediction'] == 'Seizure').sum()}")
    print(f"Healthy cases: {(results['prediction'] == 'Healthy').sum()}")
    print(f"Errors: {(results['prediction'] == 'ERROR').sum()}")


if __name__ == '__main__':
    main()
