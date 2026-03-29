"""
FastAPI wrapper for Epilepsy Detection model
Provides REST API endpoints for predictions
Run with: uvicorn api_server.py --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import io
from pathlib import Path
import os

from inference_utils import load_trained_model, get_prediction_confidence, batch_predict
from preprocessing import preprocess_eeg_file

# Initialize FastAPI app
app = FastAPI(
    title="Epilepsy Detection API",
    description="REST API for EEG-based seizure detection",
    version="1.0.0"
)

# Global model instance
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Request/Response models
class EEGSignal(BaseModel):
    """EEG signal input (100 data points)"""
    signal: List[float]
    
    class Config:
        example = {
            "signal": [0.1, 0.2, 0.15, -0.1, 0.05, None]  # 100 values
        }


class PredictionResponse(BaseModel):
    """Model prediction response"""
    prediction: str  # "Healthy" or "Seizure"
    confidence: float  # 0-1
    healthy_probability: float
    seizure_probability: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: dict


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    
    try:
        # Find latest model
        from inference_utils import find_latest_model
        model_path = find_latest_model()
        
        if model_path is None:
            print("Warning: No trained model found. Model will be loaded on first request.")
            return
        
        print(f"Loading model from: {model_path}")
        model = load_trained_model(model_path, device)
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Epilepsy Detection API",
        "version": "1.0.0",
        "status": "online",
        "device": device,
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "device": device
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(eeg_signal: EEGSignal):
    """
    Make prediction on a single EEG signal
    
    Args:
        eeg_signal: EEG signal with 100 data points
    
    Returns:
        PredictionResponse with prediction and probabilities
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(eeg_signal.signal) != 100:
            raise ValueError(f"Signal must have 100 data points, got {len(eeg_signal.signal)}")
        
        # Convert to numpy array
        signal = np.array(eeg_signal.signal, dtype=np.float32)
        
        # Check for NaN values
        if np.isnan(signal).any():
            raise ValueError("Signal contains NaN values")
        
        # Get prediction
        pred_class, confidence, probs = get_prediction_confidence(model, signal, device)
        
        return PredictionResponse(
            prediction="Seizure" if pred_class == 1 else "Healthy",
            confidence=float(confidence),
            healthy_probability=float(probs[0]),
            seizure_probability=float(probs[1])
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(signals: List[EEGSignal]):
    """
    Make predictions on multiple EEG signals
    
    Args:
        signals: List of EEG signals
    
    Returns:
        BatchPredictionResponse with all predictions and summary
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(signals) == 0:
        raise HTTPException(status_code=400, detail="No signals provided")
    
    if len(signals) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 signals per batch")
    
    try:
        predictions = []
        seizure_count = 0
        
        for signal_data in signals:
            if len(signal_data.signal) != 100:
                raise ValueError(f"Each signal must have 100 data points")
            
            signal = np.array(signal_data.signal, dtype=np.float32)
            
            if np.isnan(signal).any():
                raise ValueError("Signals contain NaN values")
            
            pred_class, confidence, probs = get_prediction_confidence(model, signal, device)
            
            pred = PredictionResponse(
                prediction="Seizure" if pred_class == 1 else "Healthy",
                confidence=float(confidence),
                healthy_probability=float(probs[0]),
                seizure_probability=float(probs[1])
            )
            
            predictions.append(pred)
            
            if pred_class == 1:
                seizure_count += 1
        
        summary = {
            "total_samples": len(signals),
            "seizure_detected": seizure_count,
            "healthy_detected": len(signals) - seizure_count,
            "seizure_percentage": round(seizure_count / len(signals) * 100, 2),
            "avg_confidence": round(np.mean([p.confidence for p in predictions]), 4)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict_from_file")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Make prediction on uploaded EEG file
    
    Args:
        file: Uploaded text file with EEG data
    
    Returns:
        Prediction with probabilities
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file
        contents = await file.read()
        
        # Parse data
        data = np.loadtxt(io.StringIO(contents.decode()), dtype=np.float32)
        
        if len(data) < 100:
            raise ValueError(f"Data must have at least 100 points, got {len(data)}")
        
        # Preprocess
        from preprocessing import preprocess_eeg_file
        segments = []
        for i in range(0, len(data) - 100, 50):
            segment = data[i:i+100]
            segments.append(segment)
        
        # Predict on all segments and average
        all_probs = []
        for segment in segments:
            pred_class, confidence, probs = get_prediction_confidence(model, segment, device)
            all_probs.append(probs[1])
        
        mean_seizure_prob = np.mean(all_probs)
        final_pred = "Seizure" if mean_seizure_prob > 0.5 else "Healthy"
        
        return {
            "filename": file.filename,
            "prediction": final_pred,
            "seizure_probability": float(mean_seizure_prob),
            "healthy_probability": float(1 - mean_seizure_prob),
            "num_segments_analyzed": len(segments),
            "confidence": float(max(mean_seizure_prob, 1 - mean_seizure_prob))
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "model_type": "TCN with Self-Attention",
        "input_shape": "(batch, 1, 100)",
        "output_shape": "(batch, 2)",
        "classes": ["Healthy", "Seizure"],
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters())
    }


@app.get("/model/performance")
async def model_performance():
    """Get cached model performance metrics"""
    
    # Try to load from evaluation results
    metrics_file = "evaluation_results/metrics.json"
    
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return {
            "status": "metrics_available",
            "metrics": metrics
        }
    else:
        return {
            "status": "no_metrics",
            "message": "Run evaluate.py to generate metrics"
        }


# Example usage documentation
@app.get("/docs")
async def docs():
    """Get API documentation"""
    return {
        "title": "Epilepsy Detection - REST API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Single EEG signal prediction",
            "POST /predict_batch": "Multiple signals prediction",
            "POST /predict_from_file": "Prediction from uploaded file",
            "GET /health": "Health check",
            "GET /model/info": "Model information",
            "GET /model/performance": "Model performance metrics",
            "GET /docs": "This documentation"
        },
        "example_request": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {
                "signal": [0.1, 0.2, 0.15, -0.1, 0.05, "...100 values..."]
            }
        },
        "example_response": {
            "prediction": "Healthy",
            "confidence": 0.95,
            "healthy_probability": 0.95,
            "seizure_probability": 0.05
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
