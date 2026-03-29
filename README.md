# 🧠 Epilepsy Detection using Deep Learning

**Production-Ready EEG-Based Seizure Detection System**

A deep learning project that detects epileptic seizures from EEG brain signals using PyTorch with Streamlit deployment.

## ✨ Features

### 🎯 Advanced Model

- **TCN with Self-Attention** architecture
- Temporal pattern recognition for EEG analysis
- High accuracy on EEG classification tasks

### 📊 Comprehensive Metrics

- **Precision & Recall** - Classification accuracy
- **F1-Score** - Balanced performance metric
- **Confusion Matrix** - Detailed classification breakdown
- **ROC-AUC** - Overall discriminative ability
- **Sensitivity & Specificity** - Medical-grade metrics

### 🌐 Production Deployment

- **Streamlit Dashboard** - Interactive web interface
- **FastAPI REST API** - Scalable backend service
- **Docker Support** - Easy containerization
- **Batch Processing** - Process multiple signals
- **Cloud Ready** - Deploy to Streamlit Cloud, AWS, GCP, Azure

### 📈 Comprehensive Evaluation

- Real-time metrics visualization
- Performance comparison plots
- Prediction confidence analysis
- Detailed evaluation reports

## 📊 Dataset

**EEG Database**:

- **Set Z**: Reference/Healthy EEG signals (files: Z\*.txt)
- **Set O**: Reference/Other EEG signals (files: O\*.txt)
- **Set N**: Normal/Seizure EEG signals (files: N\*.TXT)

Each file contains EEG data points recorded at 173.61 Hz.

### Data Organization

```
data/
├── Z/  (files: Z*.txt)
├── O/  (files: O*.txt)
└── N/  (files: N*.TXT)
```

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download from the link above and organize in `data/` folder

### 3. Train Model

```bash
python main.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

**Open:** http://localhost:8501

👉 **See [QUICKSTART.md](QUICKSTART.md) for detailed quick start guide**

## 📖 Comprehensive Guides

| Document                                   | Purpose                                        |
| ------------------------------------------ | ---------------------------------------------- |
| [QUICKSTART.md](QUICKSTART.md)             | 5-minute setup and basic usage                 |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Detailed deployment, APIs, metrics explanation |
| [README.md](README.md)                     | This file - overview and setup                 |

## 🎯 How to Run

### Training

```bash
# Simple training (uses GPU if available)
python main.py

# Force CPU
python main.py --device cpu

# Custom parameters
python train.py --epochs 100 --batch_size 32 --lr 0.01
```

### Dashboard

```bash
# Launch Streamlit web interface
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8502
```

### Evaluation

```bash
# Generate comprehensive evaluation report
python evaluate.py

# Specific model
python evaluate.py results/Z_vs_N/TCN_SA/model_fold1.pth
```

### Batch Predictions

```bash
python batch_predict.py \
    --model results/Z_vs_N/TCN_SA/model_fold1.pth \
    --input data/N \
    --output predictions.csv
```

### REST API

```bash
# Start FastAPI server
python api_server.py

# Available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📊 Dashboard Features

### 📈 Model Overview

- Architecture visualization
- Parameter count
- Device information
- Model status

### 🔬 Model Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion matrix heatmap
- ROC curve
- Prediction distribution
- Detailed metrics table

### 🧪 Test Predictions

- Real-time sample predictions
- Confidence gauge
- Probability visualization

### 📉 Performance Analysis

- KPI dashboard
- Metric comparisons
- Original metrics vs Ideal scores
- Heatmap visualizations

## 💻 Deployment Options

### 1. Local Development

```bash
streamlit run app.py
```

### 2. Docker

```bash
# Build image
docker build -t epilepsy-detection .

# Run container
docker run -p 8501:8501 epilepsy-detection

# Or with docker-compose
docker-compose up
```

### 3. Streamlit Cloud

```bash
git push to GitHub
# Go to streamlit.io/cloud
# Deploy from GitHub repo
```

### 4. Cloud Platforms

- **AWS**: EC2 + load balancer
- **GCP**: Cloud Run or App Engine
- **Azure**: Container Instances
- **DigitalOcean**: App Platform

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 🔧 API Usage

### REST API Endpoints

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ..., 0.15]}'

# Batch predictions
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[{"signal": [...]}, {"signal": [...]}]'

# File upload
curl -X POST "http://localhost:8000/predict_from_file" \
  -F "file=@eeg_data.txt"

# Model info
curl "http://localhost:8000/model/info"

# Performance metrics
curl "http://localhost:8000/model/performance"
```

### Python Client

```python
from inference_utils import load_trained_model, get_prediction_confidence
import numpy as np

# Load model
model = load_trained_model('results/Z_vs_N/TCN_SA/model_fold1.pth')

# Prepare EEG signal (100 data points)
eeg_signal = np.random.randn(100)

# Predict
pred_class, confidence, probs = get_prediction_confidence(model, eeg_signal)

print(f"Prediction: {'Seizure' if pred_class == 1 else 'Healthy'}")
print(f"Confidence: {confidence:.4f}")
```

## 📊 Understanding Metrics

### Key Performance Metrics

| Metric          | Formula          | What it means                   | Ideal |
| --------------- | ---------------- | ------------------------------- | ----- |
| **Accuracy**    | (TP+TN)/(All)    | Overall correctness             | 1.0   |
| **Precision**   | TP/(TP+FP)       | Accuracy of seizure predictions | 1.0   |
| **Recall**      | TP/(TP+FN)       | Seizure detection rate          | 1.0   |
| **F1-Score**    | 2×P×R/(P+R)      | Harmonic mean                   | 1.0   |
| **Specificity** | TN/(TN+FP)       | Healthy classification rate     | 1.0   |
| **ROC-AUC**     | Area under curve | Discriminative ability          | 1.0   |

### Confusion Matrix

```
                Predicted
             Healthy  Seizure
    Healthy    TN       FP      ← False Alarms (minimize)
Actual Seizure FN       TP      ← Missed Seizures (minimize!)
```

- **TP**: Correctly detected seizures ✓
- **TN**: Correctly identified healthy ✓
- **FP**: Healthy misclassified as seizure (false alarm)
- **FN**: Seizure missed as healthy (critical error)

### Medical Interpretation

- **High Recall** = Fewer missed seizures (critical!)
- **High Specificity** = Fewer false alarms
- **Balanced F1** = Good tradeoff between both

## 📁 Project Structure

```
Epilepsy-Detection/
├── app.py                    # 🌐 Streamlit dashboard
├── api_server.py             # 🔌 FastAPI server
├── model.py                  # 🧠 TCN-SA architecture
├── train.py                  # 📚 Training pipeline
├── evaluate.py               # 📊 Evaluation script
├── inference_utils.py        # 🔮 Inference utilities
├── batch_predict.py          # 🔄 Batch processing
├── data_loader.py            # 📦 Data loading
├── preprocessing.py          # 🔧 Signal preprocessing
├── main.py                   # 🚀 Training entry point
├── requirements.txt          # 📋 Dependencies
├── Dockerfile                # 🐳 Container config
├── docker-compose.yml        # 🐳 Compose config
├── .streamlit/config.toml    # ⚙️ Streamlit config
├── QUICKSTART.md             # 📖 Quick start
├── DEPLOYMENT_GUIDE.md       # 📖 Full guide
└── README.md                 # This file

data/
├── Z/  (Reference: Z*.txt)
├── O/  (Reference: O*.txt)
└── N/  (Normal/Seizure: N*.TXT)

results/
├── Z_vs_N/
│   └── TCN_SA/
│       ├── model_fold1.pth
│       ├── model_fold2.pth
│       └── fold_results.json
└── O_vs_N/
    └── TCN_SA/
        └── (similar structure)

evaluation_results/
├── metrics.json
└── evaluation_report.png
```

## 🔧 System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for data + trained models
- **GPU**: Optional (NVIDIA CUDA for faster training)

## 📈 Performance

| Task                      | CPU       | GPU      |
| ------------------------- | --------- | -------- |
| Data loading              | 30-60s    | 30-60s   |
| Model training (5 epochs) | 5-10 min  | 1-2 min  |
| Single prediction         | 10-50ms   | 2-5ms    |
| Batch (32 samples)        | 100-300ms | 50-100ms |
| Dashboard load            | 2-3s      | 2-3s     |

## ⚠️ Important Notes

1. **Data Privacy**: Handle EEG data according to medical/privacy regulations
2. **Clinical Use**: Not approved for clinical diagnosis - use as research tool
3. **Validation**: Always validate in your specific use case
4. **Compliance**: Ensure HIPAA/GDPR compliance when handling patient data

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional preprocessing techniques
- Model architecture variants
- Performance optimizations
- Documentation improvements
- New visualization methods

## 📚 References

- [Bonn University EEG Database](https://epileptologie-bonn.de/)
- [TCN Paper](https://arxiv.org/abs/1803.01271)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Seizure Detection Review](https://doi.org/10.1016/j.seizure.2020.06.002)

## 📝 License

[Specify your license here]

## 💬 Support

- 📖 See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed docs
- 📖 See [QUICKSTART.md](QUICKSTART.md) for quick reference
- 💻 Check code comments for implementation details
- 🐛 Report issues with detailed reproduction steps

---

**Built with ❤️ for seizure detection research**

```
Accuracy    : 0.9830 ± 0.0041 (98.30% ± 0.41%)
Sensitivity : 0.9752 ± 0.0077 (97.52% ± 0.77%)
Specificity : 0.9913 ± 0.0010 (99.13% ± 0.10%)
Precision   : 0.9911 ± 0.0014 (99.11% ± 0.14%)
F1          : 0.9830 ± 0.0039 (98.30% ± 0.39%)
Auc         : 0.9971 ± 0.0013 (99.71% ± 0.13%)
```
