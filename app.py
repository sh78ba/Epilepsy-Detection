"""
🧠 Epilepsy Detection AI - With File Upload for Predictions
"""

import streamlit as st
import numpy as np
import torch
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score
)
import pandas as pd
from data_loader import load_experiment_data, create_data_loaders
from model import create_model
from train import evaluate, set_seed
from inference_utils import load_trained_model, predict_on_data
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🧠 Epilepsy Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Metrics"

# ============================================================================
# MINIMAL STYLING
# ============================================================================
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_system_diagnostics():
    """Get system and file diagnostics"""
    import glob
    diag = {
        'device': 'GPU' if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'streamlit_version': st.__version__,
        'cwd': os.getcwd(),
        'model_files': glob.glob('results/**/*.pth', recursive=True),
        'data_dirs': [d for d in os.listdir('data') if os.path.isdir(f'data/{d}')] if os.path.exists('data') else [],
        'results_dir_exists': os.path.exists('results'),
    }
    return diag


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load trained model with enhanced diagnostics - prioritize Z_vs_N"""
    try:
        results_dir = 'results'
        model_path = None
        
        # Priority 1: Try Z_vs_N (primary experiment)
        z_vs_n_path = os.path.join(results_dir, 'Z_vs_N', 'TCN_SA', 'model_fold1.pth')
        if os.path.exists(z_vs_n_path):
            model_path = z_vs_n_path
        else:
            # Priority 2: Search for any model file
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            model_path = os.path.join(root, file)
                            break
                    if model_path:
                        break
        
        if model_path is None:
            return None, None, None, f"❌ No model found in {os.path.abspath(results_dir)}"
        
        # Verify model file exists and is readable
        if not os.path.exists(model_path):
            return None, None, None, f"❌ Model path exists in search but not accessible: {model_path}"
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_trained_model(model_path, device)
        
        if model is None:
            return None, None, None, f"❌ Failed to load model from {model_path}"
        
        return model, device, model_path, None
    except Exception as e:
        return None, None, None, f"❌ Exception: {str(e)}"


@st.cache_data
def load_metrics_data():
    """Load metrics from results file"""
    try:
        results_dir = 'results'
        metrics_file = None
        
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if 'results.json' in file:
                    metrics_file = os.path.join(root, file)
        
        if metrics_file:
            with open(metrics_file, 'r') as f:
                fold_data = json.load(f)
                return fold_data.get('test_metrics', {})
        return {}
    except:
        return {}


# ============================================================================
# PAGES
# ============================================================================
def page_metrics():
    """Metrics page"""
    st.title("📊 Model Performance Metrics")
    
    with st.spinner("Loading evaluation data..."):
        try:
            model, device, _, _ = load_model_and_data()
            data_dir = 'data'
            test_segments, test_labels, _, _ = load_experiment_data(data_dir, 'Z', 'N')
            _, test_loader = create_data_loaders(
                test_segments[:100], test_labels[:100], np.arange(100),
                test_segments[100:], test_labels[100:], np.arange(100),
                batch_size=32
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            metrics = evaluate(model, test_loader, criterion, device)
            
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
            
            cm = confusion_matrix(all_labels, all_preds)
            
            metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
            metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.2%}")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Healthy', 'Seizure'],
                    y=['Healthy', 'Seizure'],
                    color_continuous_scale="Viridis",
                    text_auto=True,
                    title="Confusion Matrix",
                )
                fig_cm.update_traces(text=cm, texttemplate="%{text}", textfont={"size": 16})
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # ROC Curve
                fpr, tpr, _ = roc_curve(all_labels, all_probs)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='#667eea', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig_roc.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            st.markdown("---")
            
            # Metrics table
            st.subheader("Detailed Metrics")
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_table = pd.DataFrame({
                'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives',
                           'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC-ROC'],
                'Value': [int(tp), int(tn), int(fp), int(fn),
                         f"{sensitivity:.4f}", f"{specificity:.4f}",
                         f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                         f"{metrics['f1']:.4f}", f"{metrics['accuracy']:.4f}", f"{metrics['auc']:.4f}"]
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


def page_predictions():
    """Real-time predictions with file upload"""
    st.title("🧪 Real-time EEG Prediction & File Upload")
    
    model, device, _, _ = load_model_and_data()
    
    # FILE UPLOAD SECTION
    with st.container():
        st.subheader("📁 Upload EEG File")
        col_upload, col_format = st.columns([3, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose EEG signal file (CSV, TXT, NPY, NPZ)",
                type=['csv', 'txt', 'npy', 'npz'],
                help="Expected: 100 EEG signal values"
            )
        
        uploaded_eeg = None
        upload_prediction = None
        
        if uploaded_file is not None:
            try:
                # Load file based on format
                if uploaded_file.name.endswith(('.csv', '.txt')):
                    uploaded_eeg = np.genfromtxt(uploaded_file, delimiter=',')
                elif uploaded_file.name.endswith('.npy'):
                    uploaded_eeg = np.load(uploaded_file)
                elif uploaded_file.name.endswith('.npz'):
                    data = np.load(uploaded_file)
                    uploaded_eeg = data[data.files[0]]
                
                # Validate and adjust shape
                if len(uploaded_eeg) > 100:
                    uploaded_eeg = uploaded_eeg[:100]
                    st.warning(f"Signal truncated to 100 samples (had {len(uploaded_eeg)} originally)")
                elif len(uploaded_eeg) < 100:
                    st.error(f"Signal too short: {len(uploaded_eeg)} samples (need 100)")
                    uploaded_eeg = None
                
                if uploaded_eeg is not None:
                    # Make prediction
                    upload_tensor = torch.FloatTensor(uploaded_eeg).unsqueeze(0).unsqueeze(0).to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(upload_tensor)
                        probs = torch.softmax(output, dim=1)
                        upload_prediction = probs[0, 1].cpu().numpy()
                    
                    st.success(f"✅ File loaded successfully! Shape: {uploaded_eeg.shape}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    st.markdown("---")
    
    # PREDICTION MODE
    if uploaded_file and uploaded_eeg is not None:
        pred_mode = st.radio(
            "Select prediction to display:",
            ["Sample Signal", "Uploaded File"],
            horizontal=True
        )
        use_uploaded = (pred_mode == "Uploaded File")
    else:
        pred_mode = "Sample Signal"
        use_uploaded = False
    
    # DISPLAY PREDICTIONS
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        if use_uploaded:
            st.subheader("📊 Uploaded EEG Signal")
            display_eeg = uploaded_eeg
            current_pred = upload_prediction
        else:
            st.subheader("📊 Sample EEG Signal")
            display_eeg = np.random.randn(100) * 0.5 + np.sin(np.linspace(0, 4*np.pi, 100)) * 0.3
            
            sample_tensor = torch.FloatTensor(display_eeg).unsqueeze(0).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(sample_tensor)
                probs = torch.softmax(output, dim=1)
                current_pred = probs[0, 1].cpu().numpy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=display_eeg,
            mode='lines',
            name='EEG Signal',
            line=dict(color='#667eea', width=3)
        ))
        fig.update_layout(
            title="EEG Waveform (100 samples)",
            xaxis_title="Time Sample",
            yaxis_title="Amplitude (μV)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if current_pred > 0.5:
            status = "⚠️ SEIZURE"
            color = "#f56565"
            status_text = "Seizure detected"
        else:
            status = "✅ HEALTHY"
            color = "#48bb78"
            status_text = "No seizure"
        
        st.markdown(f"""
        <div style="background: #f9f9f9; border-left: 5px solid {color}; padding: 20px; border-radius: 12px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">{status}</h2>
            <p style="color: #666; margin: 15px 0;">{status_text}</p>
            <h3 style="color: #667eea;">Confidence: {max(current_pred, 1-current_pred):.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Seizure Prob.", f"{current_pred:.2%}")
        st.metric("Healthy Prob.", f"{1-current_pred:.2%}")
        
        # Download Result
        if use_uploaded and uploaded_file:
            result_dict = {
                "filename": uploaded_file.name,
                "prediction": "SEIZURE" if current_pred > 0.5 else "HEALTHY",
                "confidence": float(current_pred),
                "timestamp": datetime.now().isoformat()
            }
            
            col_csv, col_json = st.columns(2)
            with col_csv:
                result_df = pd.DataFrame([result_dict])
                csv = result_df.to_csv(index=False)
                st.download_button("📥 CSV", csv, "result.csv", "text/csv")
            
            with col_json:
                json_str = json.dumps(result_dict, indent=2)
                st.download_button("📥 JSON", json_str, "result.json")
    
    st.markdown("---")
    st.subheader("📊 Model Metrics")
    
    metrics = load_metrics_data()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        with col3:
            st.metric("Recall", f"{metrics.get('sensitivity', 0):.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1', 0):.2%}")


def page_diagnostics():
    """Diagnostics page for troubleshooting"""
    st.title("🔧 System Diagnostics")
    
    st.markdown("### 🖥️ System Information")
    diag = get_system_diagnostics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Device", diag['device'])
    with col2:
        st.metric("PyTorch", diag['torch_version'])
    with col3:
        st.metric("Streamlit", diag['streamlit_version'])
    
    st.markdown("---")
    
    st.markdown("### 📁 File System")
    
    st.write(f"**Current Dir:** `{diag['cwd']}`")
    st.write(f"**Results Dir Exists:** {'✅ Yes' if diag['results_dir_exists'] else '❌ No'}")
    
    if diag['model_files']:
        st.write("**Model Files Found:**")
        for mf in diag['model_files']:
            size_mb = os.path.getsize(mf) / (1024 * 1024)
            st.write(f"  • `{mf}` ({size_mb:.2f} MB)")
    else:
        st.warning("❌ No model files found!")
    
    if diag['data_dirs']:
        st.write("**Data Directories:**")
        for dd in diag['data_dirs']:
            count = len(os.listdir(f'data/{dd}'))
            st.write(f"  • `{dd}/` ({count} files)")
    else:
        st.error("❌ No data directories found!")
    
    st.markdown("---")
    
    st.markdown("### 🤖 Model Loading")
    
    model, device, model_path, error_msg = load_model_and_data()
    
    if error_msg:
        st.error(f"❌ {error_msg}")
    else:
        st.success(f"✅ Model loaded successfully!")
        
        # Highlight which model/experiment
        if 'Z_vs_N' in model_path:
            st.info("✅ **Z vs N Model** (Healthy=Z, Seizure=N) - PRIMARY EXPERIMENT")
        elif 'O_vs_N' in model_path:
            st.warning("⚠️ **O vs N Model** (Healthy=O, Seizure=N) - Different dataset!")
        else:
            st.warning(f"❓ **Unknown Model:** {model_path}")
        
        st.write(f"**Model Path:** `{model_path}`")
        st.write(f"**Device:** {device}")
        st.write(f"**Model Type:** {type(model).__name__}")
        
        # Try to show model structure
        try:
            model_str = str(model)
            if len(model_str) > 500:
                st.write("**Model Structure (first 500 chars):**")
                st.code(model_str[:500] + "...", language="python")
            else:
                st.write("**Model Structure:**")
                st.code(model_str, language="python")
        except:
            pass
    
    st.markdown("---")
    
    st.markdown("### 📊 Data Loading Test")
    
    try:
        data_dir = 'data'
        test_segments, test_labels, _, _ = load_experiment_data(data_dir, 'Z', 'N')
        st.success(f"✅ Data loaded from `{data_dir}`")
        st.write(f"**Healthy samples:** {(test_labels == 0).sum()}")
        st.write(f"**Seizure samples:** {(test_labels == 1).sum()}")
        st.write(f"**Data shape:** {test_segments.shape}")
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### 🧪 Quick Inference Test")
    
    if model is not None:
        try:
            # Create random test data
            test_input = torch.randn(1, 1, 100).to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_input)
                probs = torch.softmax(output, dim=1)
            
            st.success("✅ Inference test passed!")
            st.write(f"**Input shape:** {test_input.shape}")
            st.write(f"**Output shape:** {output.shape}")
            st.write(f"**Predictions:** Health={probs[0,0]:.2%}, Seizure={probs[0,1]:.2%}")
        except Exception as e:
            st.error(f"❌ Inference error: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### 📝 Environment Info")
    
    env_info = {
        'Python': '3.11',
        'PyTorch': diag['torch_version'],
        'CUDA Available': 'Yes' if torch.cuda.is_available() else 'No',
        'GPU Count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    env_df = pd.DataFrame(list(env_info.items()), columns=['Parameter', 'Value'])
    st.dataframe(env_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    with st.sidebar:
        st.title("🧠 Navigation")
        
        page = st.radio(
            "Select Page:",
            ["Metrics", "Predictions", "🔧 Diagnostics"],
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page
        
        st.markdown("---")
        
        st.subheader("📡 Status")
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        st.write(f"✅ Device: {device}")
        st.write("✅ Model: Ready")
        
        st.markdown("---")
        
        st.subheader("ℹ️ Info")
        st.write("""
        **Model:** TCN + Self-Attention
        
        **Input:** EEG (100 samples)
        
        **Output:** Binary classification
        """)
    
    model, device, model_path, error_msg = load_model_and_data()
    
    if st.session_state.current_page == "🔧 Diagnostics":
        page_diagnostics()
    elif error_msg or model is None:
        st.error(f"❌ {error_msg}")
        st.warning("Please train: `python main.py`")
        st.markdown("---")
        st.markdown("**To get more info, go to → 🔧 Diagnostics**")
    elif st.session_state.current_page == "Metrics":
        page_metrics()
    elif st.session_state.current_page == "Predictions":
        page_predictions()


if __name__ == "__main__":
    main()
