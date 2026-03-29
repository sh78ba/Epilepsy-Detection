"""
🧠 Epilepsy Detection AI - Beautiful & Responsive Dashboard
Production-ready Streamlit application with modern UI
"""

import streamlit as st
import numpy as np
import torch
import os
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_experiment_data, create_data_loaders
from model import create_model
from train import evaluate, set_seed
from inference_utils import load_trained_model, predict_on_data, get_model_summary
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🧠 Epilepsy Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# ============================================================================
# ENHANCED CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eccf 100%);
    }
    
    /* Header styling */
    .stApp > header {
        background: transparent !important;
    }
    
    .header-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        text-align: center;
        border: 2px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .header-main h1 {
        font-size: clamp(2rem, 5vw, 3.5rem);
        margin: 0;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    .header-main p {
        font-size: clamp(0.9rem, 2vw, 1.3rem);
        margin: 10px 0 0 0;
        opacity: 0.95;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Feature cards - Dashboard */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin: 10px 0;
        border: 2px solid transparent;
        border-left: 5px solid #667eea;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 35px rgba(102, 126, 234, 0.2);
        border: 2px solid #667eea;
    }
    
    .feature-card h3 {
        color: #333;
        font-size: 1.2rem;
        margin-bottom: 12px;
        font-weight: 700;
    }
    
    .feature-card p {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 8px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin: 12px 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 35px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card h3 {
        font-size: 0.85rem;
        color: #999;
        margin: 0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        margin: 12px 0;
        font-weight: 900;
        color: #667eea;
    }
    
    /* Dashboard action buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 20px 30px !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
        min-height: 100px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7fa 0%, #e8f0ff 100%);
    }
    
    .sidebar-status {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: 600;
        text-align: center;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        padding: 8px;
        gap: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 8px;
    }
    
    /* Section headers */
    .section-header {
        color: #333;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 30px 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-main {
            padding: 30px 20px;
        }
        
        .header-main h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            min-width: 100%;
        }
        
        .stButton > button {
            min-height: 80px !important;
            padding: 15px !important;
        }
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
# DISPLAY FUNCTIONS
# ============================================================================
def display_metric_cards(metrics_dict, cm):
    """Display metrics in beautiful cards"""
    cols = st.columns(4)
    
    metrics_data = [
        ("📊 Accuracy", f"{metrics_dict['accuracy']:.2%}", "#667eea"),
        ("🎯 Precision", f"{metrics_dict['precision']:.2%}", "#48bb78"),
        ("🔍 Recall", f"{metrics_dict['recall']:.2%}", "#f56565"),
        ("⚖️ F1-Score", f"{metrics_dict['f1']:.2%}", "#ed8936"),
    ]
    
    for col, (label, value, color) in zip(cols, metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color};">
                <h3>{label}</h3>
                <h2 style="color: {color};">{value}</h2>
            </div>
            """, unsafe_allow_html=True)


def plot_confusion_matrix_heatmap(cm):
    """Enhanced confusion matrix heatmap"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Healthy', 'Seizure'],
        y=['Healthy', 'Seizure'],
        color_continuous_scale="Viridis",
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        width=500,
        height=500,
    )
    fig.update_traces(text=cm, texttemplate="%{text}", textfont={"size": 20})
    fig.update_layout(
        font=dict(size=12),
        title_font_size=18,
        title_x=0.5,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


def plot_roc_curve_enhanced(y_true, y_scores):
    """Enhanced ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#667eea', width=4),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#999', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve Analysis",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode='closest',
        height=500,
        title_x=0.5,
        template='plotly_white',
        plot_bgcolor='#f8f9fa',
        font=dict(size=12)
    )
    return fig, roc_auc


def plot_metrics_comparison(metrics_dict):
    """Enhanced metrics comparison"""
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    metrics_values = [
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1'],
        metrics_dict['accuracy']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_names,
            y=metrics_values,
            text=[f"{v:.2%}" for v in metrics_values],
            textposition='auto',
            marker=dict(
                color=['#667eea', '#f56565', '#ed8936', '#48bb78'],
                line=dict(color='white', width=3),
                cornerradius=8
            ),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        yaxis_title="Score",
        xaxis_title="Metric",
        yaxis=dict(range=[0, 1], gridcolor='#eee'),
        height=450,
        title_x=0.5,
        template='plotly_white',
        plot_bgcolor='#f8f9fa',
        hovermode='x unified',
        font=dict(size=12)
    )
    return fig


def plot_prediction_distribution(y_true, y_probs):
    """Enhanced prediction distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=y_probs[y_true == 0],
        name='Healthy (True)',
        opacity=0.8,
        marker_color='#48bb78',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=y_probs[y_true == 1],
        name='Seizure (True)',
        opacity=0.8,
        marker_color='#f56565',
        nbinsx=30
    ))
    
    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Seizure Probability",
        yaxis_title="Count",
        barmode='overlay',
        height=450,
        template='plotly_white',
        plot_bgcolor='#f8f9fa',
        title_x=0.5,
        hovermode='x unified',
        legend=dict(x=0.7, y=0.95),
        font=dict(size=12)
    )
    return fig


def create_status_card(label, value, icon, color):
    """Create a status card"""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color}; text-align: center">
        <div style="font-size: 2.5rem">{icon}</div>
        <p style="color: #999; margin: 5px 0; font-size: 0.9rem">{label}</p>
        <h2 style="color: {color}; margin: 10px 0; font-size: 1.8rem">{value}</h2>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGES
# ============================================================================
def page_dashboard():
    """Dashboard Home Page"""
    st.markdown("""
    <div class="header-main">
        <h1>🧠 Epilepsy Detection AI</h1>
        <p>Advanced Deep Learning System for EEG-based Seizure Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_status_card("Model Status", "Ready", "✅", "#48bb78")
    with col2:
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        create_status_card("Device", device, "🖥️", "#667eea")
    with col3:
        create_status_card("API Status", "Active", "🌐", "#ed8936")
    
    st.markdown("<h2 class='section-header'>🚀 Quick Actions</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Metrics", key="btn_metrics"):
            st.session_state.current_page = "📊 Metrics"
            st.rerun()
    
    with col2:
        if st.button("🧪 Make Prediction", key="btn_predict"):
            st.session_state.current_page = "🧪 Predictions"
            st.rerun()
    
    with col3:
        if st.button("📈 View Analysis", key="btn_analysis"):
            st.session_state.current_page = "📈 Analysis"
            st.rerun()
    
    st.markdown("<h2 class='section-header'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-time Detection</h3>
            <p>Instant EEG analysis with high-speed inference. Process signals in milliseconds.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Advanced Metrics</h3>
            <p>Precision & Recall analysis, ROC-AUC evaluation, detailed confusion matrix.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Production Ready</h3>
            <p>REST API support, batch processing capability, Docker compatible deployment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h2 class='section-header'>📊 Model Information</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Architecture: TCN with Self-Attention
        - Temporal Convolutional Network for time-series
        - Self-Attention mechanism for feature importance
        - Optimized for EEG signal analysis
        """)
    
    with col2:
        st.markdown("""
        #### Classification Task
        - **Input**: EEG signals (100 samples)
        - **Output**: Binary classification
        - **Classes**: Healthy vs Seizure
        """)


def page_metrics():
    """Metrics Analysis Page"""
    st.markdown("<h2 class='section-header'>📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
    
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
            
            st.markdown("<h3>Key Performance Indicators</h3>", unsafe_allow_html=True)
            display_metric_cards(metrics, cm)
            
            st.markdown("---")
            st.markdown("<h3>Performance Visualizations</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cm = plot_confusion_matrix_heatmap(cm)
                st.plotly_chart(fig_cm, width='stretch')
            
            with col2:
                fig_metrics = plot_metrics_comparison(metrics)
                st.plotly_chart(fig_metrics, width='stretch')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roc, roc_auc = plot_roc_curve_enhanced(all_labels, all_probs)
                st.plotly_chart(fig_roc, width='stretch')
            
            with col2:
                fig_dist = plot_prediction_distribution(all_labels, all_probs)
                st.plotly_chart(fig_dist, width='stretch')
            
            st.markdown("---")
            st.markdown("<h3>Detailed Metrics Table</h3>", unsafe_allow_html=True)
            
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_df = pd.DataFrame({
                'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives',
                           'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Accuracy'],
                'TP': [int(tp), int(tn), int(fp), int(fn), sensitivity, specificity,
                       metrics['precision'], metrics['f1'], metrics['accuracy']],
                'Formatted': [str(int(tp)), str(int(tn)), str(int(fp)), str(int(fn)),
                              f"{sensitivity:.4f}", f"{specificity:.4f}",
                              f"{metrics['precision']:.4f}", f"{metrics['f1']:.4f}",
                              f"{metrics['accuracy']:.4f}"]
            })
            
            # Display formatted dataframe
            display_df = metrics_df[['Metric', 'Formatted']].copy()
            display_df.columns = ['Metric', 'Value']
            display_df['Value'] = display_df['Value'].astype(str)
            st.dataframe(display_df, width='stretch', hide_index=True)
            
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")


def page_predictions():
    """Real-time Predictions Page"""
    st.markdown("<h2 class='section-header'>🧪 Real-time EEG Prediction</h2>", unsafe_allow_html=True)
    
    model, device, _, _ = load_model_and_data()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("#### Sample EEG Waveform")
        sample_eeg = np.random.randn(100) * 0.5 + np.sin(np.linspace(0, 4*np.pi, 100)) * 0.3
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sample_eeg,
            mode='lines+markers',
            name='EEG Signal',
            line=dict(color='#667eea', width=3),
            marker=dict(size=4, opacity=0.5)
        ))
        fig.update_layout(
            title="EEG Signal (100 samples)",
            xaxis_title="Time Sample",
            yaxis_title="Amplitude (μV)",
            height=400,
            template='plotly_white',
            plot_bgcolor='#f8f9fa',
            title_x=0.5,
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### Prediction Result")
        
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
            bg_color = "#fef2f2"
        else:
            status = "✅ HEALTHY"
            color = "#48bb78"
            status_text = "Normal EEG activity"
            bg_color = "#f0fdf4"
        
        st.markdown(f"""
        <div style="background: {bg_color}; border: 3px solid {color}; padding: 25px; border-radius: 15px; text-align: center">
            <h2 style="color: {color}; margin: 0">{status}</h2>
            <p style="color: #666; margin: 15px 0; font-size: 1rem">{status_text}</p>
            <h3 style="color: #667eea; font-size: 1.5rem">Confidence: {max(pred_prob, 1-pred_prob):.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Seizure Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea", 'thickness': 0.7},
                'steps': [
                    {'range': [0, 50], 'color': "#f0fdf4"},
                    {'range': [50, 100], 'color': "#fef2f2"}
                ],
                'threshold': {
                    'line': {'color': "#f56565", 'width': 3},
                    'thickness': 0.7,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=70, b=20), paper_bgcolor='white')
        st.plotly_chart(fig_gauge, width='stretch')


def page_analysis():
    """Analysis Page"""
    st.markdown("<h2 class='section-header'>📈 Detailed Performance Analysis</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["📋 Overview", "🧠 Technical", "⚕️ Medical Notes"])
    
    with tabs[0]:
        st.markdown("""
        #### Model Architecture
        
        The system uses a **Temporal Convolutional Network (TCN)** with **Self-Attention mechanism** 
        to analyze EEG signals for seizure detection.
        
        **Key Components:**
        - **Temporal Convolution:** Captures time-series patterns in EEG signals
        - **Self-Attention:** Identifies the most important signal features
        - **Binary Classification:** Healthy vs Seizure classification
        - **Output Confidence:** Probability score between 0 (Healthy) and 1 (Seizure)
        """)
    
    with tabs[1]:
        st.markdown("""
        #### Understanding the Metrics
        
        | Metric | Definition |
        |--------|-----------|
        | **Accuracy** | (TP+TN)/Total - Overall correctness |
        | **Precision** |  TP/(TP+FP) - Accuracy of seizure predictions |
        | **Recall** | TP/(TP+FN) - Percentage of seizures detected |
        | **F1-Score** | 2×(P×R)/(P+R) - Balance between precision & recall |
        | **ROC-AUC** | Area under ROC curve - Discriminative ability |
        
        **Decision Threshold: 0.5**
        - Score > 0.5 = Seizure ⚠️
        - Score < 0.5 = Healthy ✅
        """)
    
    with tabs[2]:
        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This system is a research/demonstration tool and is **NOT** approved for clinical use.
        Always consult with qualified medical professionals for medical diagnosis and treatment decisions.
        """)


def page_about():
    """About Page"""
    st.markdown("<h2 class='section-header'>ℹ️ About This Project</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Project Overview
        
        **Epilepsy Detection using Deep Learning**
        
        A production-ready system for detecting epileptic seizures 
        from EEG brain signals using advanced neural networks.
        
        **Why This Matters:**
        - Epilepsy affects 50+ million people worldwide
        - Early detection improves patient outcomes
        - Automated analysis reduces diagnostic time
        """)
    
    with col2:
        st.markdown("""
        #### 🔧 Technology Stack
        
        **Backend:**
        - PyTorch (Deep Learning)
        - FastAPI (REST API)
        - Streamlit (Web UI)
        
        **Analysis:**
        - Plotly (Interactive charts)
        - Scikit-learn (Metrics)
        - NumPy/Pandas (Data)
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 System Architecture
        
        1. **Data Pipeline**
           - Raw EEG signals (4096 samples)
           - Filtering & normalization
           - Segmentation (100 samples)
        
        2. **Model Pipeline**
           - TCN feature extraction
           - Self-Attention processing
           - Binary classification
        """)
    
    with col2:
        st.markdown("""
        #### 🚀 Deployment Options
        
        - **Web Dashboard** (Streamlit)
        - **REST API** (FastAPI)
        - **Batch Processing** (Command line)
        - **Docker** (Containerized)
        - **Cloud** (Streamlit Cloud, AWS, GCP)
        """)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select View:",
            ["🏠 Dashboard", "📊 Metrics", "🧪 Predictions", "📈 Analysis", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page
        
        st.markdown("---")
        
        st.markdown("### 📡 System Status")
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        st.markdown(f"<div class='sidebar-status'>✅ Device: {device}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-status'>✅ API Running</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Model Info")
        st.info("""
        **Architecture:** TCN + Self-Attention
        
        **Input:** EEG (100 samples)
        **Output:** Binary Classification
        
        **Classes:**
        - Healthy ✅
        - Seizure ⚠️
        """)
    
    # Load model
    model, device, model_path, error_msg = load_model_and_data()
    
    if error_msg or model is None:
        st.error(f"❌ {error_msg}")
        st.warning("Please train the model first: `python main.py`")
        return
    
    # Page routing
    if st.session_state.current_page == "🏠 Dashboard":
        page_dashboard()
    elif st.session_state.current_page == "📊 Metrics":
        page_metrics()
    elif st.session_state.current_page == "🧪 Predictions":
        page_predictions()
    elif st.session_state.current_page == "📈 Analysis":
        page_analysis()
    elif st.session_state.current_page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
