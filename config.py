"""
Configuration file for model training and inference
Centralized settings for easy modification
"""

import os

# ===== HARDWARE CONFIGURATION =====
DEVICE = 'cuda'  # 'cuda' or 'cpu'
NUM_WORKERS = 4  # Data loading workers

# ===== DATA CONFIGURATION =====
DATA_DIR = 'data'
SAMPLE_RATE_ORIGINAL = 173.61  # Hz
SAMPLE_RATE_TARGET = 100  # Hz
SEGMENT_LENGTH = 100  # data points
APPLY_FILTERING = True

# ===== MODEL CONFIGURATION =====
MODEL_CONFIG = {
    'input_channels': 1,
    'input_length': 100,
    'num_tcn_blocks': 2,
    'tcn_channels': 128,
    'kernel_size': 4,
    'num_classes': 2,
    'tcn_dropout': 0.2,
    'sa_dropout': 0.1,
    'fc_dropout': 0.3
}

# ===== TRAINING CONFIGURATION =====
TRAINING_CONFIG = {
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'early_stopping_patience': 20,
    'n_folds': 3,
    'random_state': 42
}

# ===== EVALUATION CONFIGURATION =====
EVALUATION_CONFIG = {
    'train_split': 0.8,
    'test_split': 0.2,
    'random_state': 42
}

# ===== DIRECTORIES =====
RESULTS_DIR = 'results'
EVALUATION_DIR = 'evaluation_results'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== STREAMLIT CONFIGURATION =====
STREAMLIT_CONFIG = {
    'page_title': 'Epilepsy Detection Dashboard',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# ===== DATA AUGMENTATION (if needed) =====
AUGMENTATION_CONFIG = {
    'enabled': False,
    'noise_std': 0.01,
    'shift_range': 5,
    'scale_range': 0.1
}

# ===== DISPLAY CONFIGURATION =====
DISPLAY_CONFIG = {
    'font_size': 12,
    'figure_dpi': 300,
    'colormap': 'viridis'
}

# ===== API CONFIGURATION =====
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info'
}

# ===== METRICS THRESHOLDS =====
METRICS_THRESHOLDS = {
    'seizure_probability': 0.5,  # Decision threshold
    'min_confidence': 0.6,  # Minimum acceptable confidence
    'good_accuracy': 0.90,
    'good_f1': 0.85
}

# ===== CLASS LABELS =====
CLASS_LABELS = {
    0: 'Healthy',
    1: 'Seizure'
}

CLASS_NAMES = ['Healthy', 'Seizure']

# ===== EXPERIMENT CONFIGURATIONS =====
EXPERIMENTS = {
    'Z_vs_N': {
        'name': 'Z vs N Dataset',
        'healthy_set': 'Z',
        'seizure_set': 'N',
        'description': 'Comparison between Z and N EEG datasets'
    },
    'O_vs_N': {
        'name': 'O vs N Dataset',
        'healthy_set': 'O',
        'seizure_set': 'N',
        'description': 'Comparison between O and N EEG datasets'
    }
}

# ===== LOGGING CONFIGURATION =====
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# ===== RANDOM SEED FOR REPRODUCIBILITY =====
RANDOM_SEED = 42
