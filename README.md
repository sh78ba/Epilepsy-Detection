# 🧠 Epilepsy Detection using Deep Learning

A deep learning project that detects epileptic seizures from EEG brain signals using PyTorch.

## What does this do?

This project trains a neural network to identify epileptic seizures by analyzing EEG (brain wave) recordings. It can distinguish between:
- Normal brain activity (healthy person)
- Seizure activity (epileptic episode)

The model achieves over 95% accuracy on test data!  

## 📊 Dataset

We use the **Bonn University EEG Database** which contains brain recordings from:
- **Set A**: Healthy people with eyes open (100 files)
- **Set B**: Healthy people with eyes closed (100 files)
- **Set E**: People having seizures (100 files)

Each file contains 4096 data points recorded over 23.6 seconds.

**Download Link**: [Bonn EEG Dataset](https://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3)

### How to organize your data

After downloading, your folder should look like this:

```
epilepsyDetection/
├── data/
│   ├── A/          (files: Z001.txt to Z100.txt)
│   ├── B/          (files: O001.txt to O100.txt)
│   └── E/          (files: S001.txt to S100.txt)
├── main.py
├── model.py
└── ... (other Python files)
```

## 🚀 Installation

### What you need

- Python 3.8 or higher
- A computer (GPU optional but makes training faster)

### Install dependencies

1. Open terminal/command prompt

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset (see above) and put files in the `data` folder

That's it! You're ready to run.

## 🎯 How to Run

### Simple command (runs everything):

```bash
python main.py
```

This will:
- Train on both experiments (A vs E and B vs E)
- Take about 1-2 hours with GPU (longer on CPU)
- Save results in the `results/` folder


### If you don't have a GPU:

```bash
python main.py --device cpu
```

(Warning: This will be slower!)


## 📈 Results

After training, you'll find results in the `results/` folder:

```
results/
├── A_vs_E/          (Healthy eyes open vs Seizure)
│   └── TCN_SA/
│       ├── model_fold1.pth
│       ├── model_fold2.pth
│       ├── model_fold3.pth
│       └── final_results.json
└── B_vs_E/          (Healthy eyes closed vs Seizure)
    └── TCN_SA/
        └── (same files)
```


```
Accuracy    : 0.9830 ± 0.0041 (98.30% ± 0.41%)
Sensitivity : 0.9752 ± 0.0077 (97.52% ± 0.77%)
Specificity : 0.9913 ± 0.0010 (99.13% ± 0.10%)
Precision   : 0.9911 ± 0.0014 (99.11% ± 0.14%)
F1          : 0.9830 ± 0.0039 (98.30% ± 0.39%)
Auc         : 0.9971 ± 0.0013 (99.71% ± 0.13%)
```