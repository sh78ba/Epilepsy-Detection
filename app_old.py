"""
🧠 Epilepsy Detection AI - Simplified Dashboard
"""

import streamlit as st
import numpy as np
import torch
import os
import json
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
# CACHE FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load trained model"""
    try:
        results_dir = 'results'
        model_path = None
        
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.pth'):
                    model_path = os.path.join(root, file)
                    break
        
        if model_path is None:
            return None, None, None, "No trained model found"
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_trained_model(model_path, device)
        
        return model, device, model_path, None
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"


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
    """Real-time predictions"""
    st.title("🧪 Real-time EEG Prediction")
    
    model, device, _, _ = load_model_and_data()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Sample EEG Signal")
        # Generate sample signal
        sample_eeg = np.random.randn(100) * 0.5 + np.sin(np.linspace(0, 4*np.pi, 100)) * 0.3
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sample_eeg,
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
        st.subheader("Prediction Result")
        
        sample_tensor = torch.FloatTensor(sample_eeg).unsqueeze(0).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(sample_tensor)
            probs = torch.softmax(output, dim=1)
            pred_prob = probs[0, 1].cpu().numpy()
        
        if pred_prob > 0.5:
            status = "⚠️ SEIZURE DETECTED"
            color = "#f56565"
            status_text = "Seizure activity detected"
        else:
            status = "✅ HEALTHY"
            color = "#48bb78"
            status_text = "Normal EEG activity"
        
        st.markdown(f"""
        <div style="background: #f9f9f9; border-left: 5px solid {color}; padding: 20px; border-radius: 12px; text-align: center;">
            <h2 style="color: {color}; margin: 0;">{status}</h2>
            <p style="color: #666; margin: 15px 0;">{status_text}</p>
            <h3 style="color: #667eea;">Confidence: {max(pred_prob, 1-pred_prob):.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Seizure Probability", f"{pred_prob:.2%}")
        st.metric("Healthy Probability", f"{1-pred_prob:.2%}")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🧠 Navigation")
        
        page = st.radio(
            "Select Page:",
            ["Metrics", "Predictions"],
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
    
    # Load model
    model, device, model_path, error_msg = load_model_and_data()
    
    if error_msg or model is None:
        st.error(f"❌ {error_msg}")
        st.warning("Please train the model first: `python main.py`")
        return
    
    # Page routing
    if st.session_state.current_page == "Metrics":
        page_metrics()
    elif st.session_state.current_page == "Predictions":
        page_predictions()


if __name__ == "__main__":
    main()
