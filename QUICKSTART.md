# Quick Start Guide - Epilepsy Detection

## 5-Minute Setup

### 1. Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### 2. Organize Data (1 min)

Place your EEG data in the following structure:

```
data/
├── Z/  (Z*.txt files)
├── O/  (O*.txt files)
└── N/  (N*.TXT files)
```

### 3. Train Model (depends on hardware - 5-30 min)

```bash
python main.py
```

### 4. Launch Dashboard (1 sec)

```bash
streamlit run app.py
```

**Done!** Open http://localhost:8501

---

## What Can You Do?

### 📊 View Model Dashboard

The Streamlit app shows:

- ✅ Model architecture details
- ✅ Training/testing metrics
- ✅ Precision, Recall, F1-Score
- ✅ Confusion Matrix visualization
- ✅ ROC curves and performance analysis
- ✅ Real-time predictions

### 🧪 Make Predictions

```python
from inference_utils import load_trained_model, get_prediction_confidence
import numpy as np

# Load model
model = load_trained_model('results/Z_vs_N/TCN_SA/model_fold1.pth')

# Create dummy EEG signal
eeg_signal = np.random.randn(100)

# Get prediction
prediction, confidence, probabilities = get_prediction_confidence(model, eeg_signal)

print(f"Prediction: {'Seizure' if prediction == 1 else 'Healthy'}")
print(f"Confidence: {confidence:.4f}")
```

### 📈 Evaluate Model

```bash
# Generate detailed evaluation report
python evaluate.py results/Z_vs_N/TCN_SA/model_fold1.pth
```

This creates:

- `evaluation_results/metrics.json` - All metrics
- `evaluation_results/evaluation_report.png` - Visualizations

### 🔋 Batch Predictions

```bash
# Predict on multiple files
python batch_predict.py \
    --model results/Z_vs_N/TCN_SA/model_fold1.pth \
    --input data/N \
    --output predictions.csv
```

---

## Understanding Metrics

### Basic Metrics

| Metric        | What it means                                       |
| ------------- | --------------------------------------------------- |
| **Accuracy**  | % of correct predictions                            |
| **Precision** | When model predicts seizure, how often is it right? |
| **Recall**    | What % of actual seizures does model detect?        |
| **F1-Score**  | Balance between precision and recall                |

### Medical Metrics

| Metric                   | Why it matters                                  |
| ------------------------ | ----------------------------------------------- |
| **Sensitivity (Recall)** | Don't miss seizures!                            |
| **Specificity**          | Avoid false alarms                              |
| **ROC-AUC**              | Overall model quality (0.5=random, 1.0=perfect) |

### Confusion Matrix

```
                Predicted
             Healthy  Seizure
Real Healthy    TN       FP
Real Seizure    FN       TP

TP = Correct seizure detection
TN = Correct healthy classification
FP = False alarm (bad)
FN = Missed seizure (worst!)
```

---

## Deployment Options

### 🏠 Local Machine

```bash
streamlit run app.py
```

### 🐳 Docker

```bash
docker-compose up
# Visit http://localhost:8501
```

### ☁️ Cloud (Streamlit Cloud)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy!

### 🔧 As API

```bash
pip install fastapi uvicorn
python api_example.py
# Use endpoints for predictions
```

---

## Common Issues & Solutions

### ❌ "No data files found"

```bash
# Check data structure
ls data/Z/  # Should show Z*.txt files
ls data/O/  # Should show O*.txt files
ls data/N/  # Should show N*.TXT files
```

### ❌ GPU out of memory

```python
# In data_loader.py, change:
batch_size = 16  # was 32
```

### ❌ Streamlit not starting

```bash
# Clear cache and restart
rm -rf ~/.streamlit
streamlit run app.py --logger.level=debug
```

---

## Project Files Overview

| File                 | Purpose                         |
| -------------------- | ------------------------------- |
| `app.py`             | 🌐 Streamlit web dashboard      |
| `model.py`           | 🧠 TCN-SA neural network        |
| `train.py`           | 📚 Training loop with metrics   |
| `evaluate.py`        | 📊 Model evaluation & reporting |
| `inference_utils.py` | 🔮 Prediction utilities         |
| `batch_predict.py`   | 🔄 Batch processing             |
| `data_loader.py`     | 📦 Data loading pipeline        |
| `preprocessing.py`   | 🔧 EEG signal preprocessing     |

---

## Next Steps

1. **Train Model** → `python main.py`
2. **Launch Dashboard** → `streamlit run app.py`
3. **Explore Metrics** → Check "Model Evaluation" tab
4. **Make Predictions** → Use "Test Predictions" tab
5. **Deploy** → Use Docker or Streamlit Cloud

---

## Need Help?

📖 See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed documentation

📝 Check [README.md](README.md) for project description

💻 View code comments for implementation details

---

## Performance Expectations

| Component                 | Time      |
| ------------------------- | --------- |
| Data loading              | 30-60 sec |
| Model training (5 epochs) | 5-10 min  |
| Single prediction (CPU)   | 10-50 ms  |
| Single prediction (GPU)   | 2-5 ms    |
| Dashboard load            | 2-3 sec   |

---

**Happy predicting! 🚀**
