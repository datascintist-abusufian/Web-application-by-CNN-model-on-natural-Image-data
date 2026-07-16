import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from scipy import ndimage
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced CIFAR-10 Research Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .research-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #00a6ed;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00a6ed;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .prediction-high {
        padding: 1rem;
        background: #d4edda;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .prediction-low {
        padding: 1rem;
        background: #f8d7da;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background: #d1ecf1;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASS_INDICES = {name: idx for idx, name in enumerate(CLASS_NAMES)}
COLORS = px.colors.qualitative.Set3

MODEL_URLS = {
    'CNN': "https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_cnn.h5",
    'ResNet50': "https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_resnet50.h5",
    'VGG16': "https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_vgg16.h5",
    'MobileNetV2': "https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_mobilenetv2.h5"
}

BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/"
IMAGE_SIZE = 224

# ============================================================================
# SESSION STATE
# ============================================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = 'CNN'
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'gradcam_available' not in st.session_state:
        st.session_state.gradcam_available = False
    if 'feature_extraction' not in st.session_state:
        st.session_state.feature_extraction = None

initialize_session_state()

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model_from_url(model_type='CNN'):
    """Load the model from URL or local file"""
    try:
        model_url = MODEL_URLS.get(model_type, MODEL_URLS['CNN'])
        model_name = model_url.split('/')[-1]
        model_path = f'models/{model_name}'
        
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(model_path):
            with st.spinner(f'📥 Downloading {model_type} model... This may take a few minutes...'):
                response = requests.get(model_url, stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    progress_bar = st.progress(0)
                    downloaded = 0
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(min(progress, 1.0))
                    
                    progress_bar.empty()
                    st.success(f"✅ Model downloaded successfully!")
                else:
                    st.error(f"Failed to download model. Status code: {response.status_code}")
                    return None
        
        model = load_model(model_path)
        st.session_state.model_type = model_type
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image, model_type='CNN'):
    """Preprocess image based on model type"""
    try:
        if model_type in ['ResNet50', 'VGG16', 'MobileNetV2']:
            img_resized = image.resize((224, 224))
        else:
            img_resized = image.resize((32, 32))
        
        img_array = np.array(img_resized)
        
        if model_type == 'CNN':
            img_array = img_array / 255.0
        elif model_type == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
            img_array = resnet_preprocess(img_array.astype(np.float32))
        elif model_type == 'VGG16':
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
            img_array = vgg_preprocess(img_array.astype(np.float32))
        elif model_type == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
            img_array = mobilenet_preprocess(img_array.astype(np.float32))
        
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_prediction_confidence(predictions):
    """Create enhanced prediction confidence visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Prediction Confidence by Class', 'Confidence Distribution'],
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Bar(
            x=CLASS_NAMES,
            y=predictions[0] * 100,
            marker_color=COLORS,
            text=[f"{pred:.1f}%" for pred in predictions[0] * 100],
            textposition='outside',
            name='Confidence'
        ),
        row=1, col=1
    )
    
    sorted_preds = np.sort(predictions[0])[::-1] * 100
    fig.add_trace(
        go.Scatter(
            x=list(range(len(sorted_preds))),
            y=sorted_preds,
            mode='lines+markers',
            name='Sorted Confidence',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Class", row=1, col=1)
    fig.update_yaxes(title_text="Confidence (%)", row=1, col=1)
    fig.update_xaxes(title_text="Rank", row=2, col=1)
    fig.update_yaxes(title_text="Confidence (%)", row=2, col=1)
    
    return fig

def plot_confidence_entropy(predictions):
    """Calculate and plot confidence entropy"""
    probs = predictions[0]
    entropy_value = entropy(probs, base=2)
    max_entropy = np.log2(len(CLASS_NAMES))
    normalized_entropy = entropy_value / max_entropy
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=[probs],
        y=['Confidence'],
        x=CLASS_NAMES,
        colorscale='RdBu',
        zmin=0,
        zmax=1,
        text=[[f"{p:.2%}" for p in probs]],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'Confidence Heatmap (Entropy: {entropy_value:.3f} bits, Normalized: {normalized_entropy:.2%})',
        height=200,
        xaxis_title="Class",
        yaxis_title=""
    )
    
    return fig, entropy_value, normalized_entropy

def plot_feature_space(features, labels=None, method='PCA'):
    """Visualize feature space using PCA or t-SNE"""
    if features is None:
        return None
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Feature Space'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 't-SNE Visualization of Feature Space'
    
    features_2d = reducer.fit_transform(features)
    
    fig = go.Figure()
    
    if labels is not None:
        for i, class_name in enumerate(CLASS_NAMES):
            mask = labels == i
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=features_2d[mask, 0],
                    y=features_2d[mask, 1],
                    mode='markers',
                    name=class_name,
                    marker=dict(size=8, opacity=0.7),
                    text=[class_name] * np.sum(mask)
                ))
    else:
        fig.add_trace(go.Scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            mode='markers',
            marker=dict(size=8, color='#667eea', opacity=0.7)
        ))
    
    fig.update_layout(
        title=title,
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    """Create confusion matrix visualization with proper error handling"""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            return None
        
        if len(y_true) != len(y_pred):
            return None
        
        all_classes = sorted(set(y_true) | set(y_pred))
        class_names = [CLASS_NAMES[i] for i in all_classes if i < len(CLASS_NAMES)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title='Confusion Matrix'
        )
        
        fig.update_layout(height=500)
        return fig
        
    except Exception as e:
        st.warning(f"Could not create confusion matrix: {str(e)}")
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>🧠 Advanced CIFAR-10 Research Classifier</h1>
        <p>Multi-Model Deep Learning Analysis for Research Applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Research Controls")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Architecture",
            ['CNN', 'ResNet50', 'VGG16', 'MobileNetV2'],
            help="Different architectures for comparison"
        )
        
        # Load model
        if st.button("🔄 Load Model", use_container_width=True):
            with st.spinner(f"Loading {model_type} model..."):
                st.session_state.model = load_model_from_url(model_type)
                st.session_state.model_type = model_type
                if st.session_state.model:
                    st.success(f"✅ {model_type} model loaded!")
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("🔬 Analysis Options")
        show_feature_space = st.checkbox("Show Feature Space", value=True)
        show_confusion = st.checkbox("Show Confusion Matrix", value=False)
        show_gradcam = st.checkbox("Enable Grad-CAM", value=False)
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Information")
        st.info(f"""
        **Current Model:** {st.session_state.model_type if st.session_state.model else 'Not loaded'}
        **Classes:** 10
        **Input Size:** {'224x224' if model_type != 'CNN' else '32x32'}
        **Dataset:** CIFAR-10
        """)
        
        if st.session_state.prediction_history:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
            correct = sum([1 for p in st.session_state.prediction_history if p.get('correct', False)])
            if len(st.session_state.prediction_history) > 0:
                st.metric("Accuracy", f"{(correct/len(st.session_state.prediction_history))*100:.1f}%")
    
    # Check model
    if st.session_state.model is None:
        st.warning("⚠️ Please load a model from the sidebar to begin analysis.")
        st.info("💡 Click the 'Load Model' button to download and load a pre-trained model.")
        return
    
    # Single image analysis
    st.subheader("🔍 Image Selection & Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Class selection
        selected_class = st.selectbox(
            "Select image class for analysis:",
            CLASS_NAMES,
            format_func=lambda x: x.capitalize(),
            key='class_select'
        )
        
        try:
            image_url = f"{BASE_IMAGE_URL}cifar_image_{selected_class.lower()}_1.png"
            response = requests.get(image_url)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Display image - FIXED: use width instead of use_container_width
                st.image(image, caption=f"Sample {selected_class.capitalize()} Image", width=None)
                
                # Upload option
                uploaded_file = st.file_uploader(
                    "Or upload your own image",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", width=None)
                
                # Analyze button
                if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Preprocess
                        processed = preprocess_image(image, st.session_state.model_type)
                        
                        if processed is not None:
                            # Predict
                            start_time = time.time()
                            predictions = st.session_state.model.predict(processed, verbose=0)
                            inference_time = time.time() - start_time
                            
                            predicted_idx = np.argmax(predictions)
                            predicted_class = CLASS_NAMES[predicted_idx]
                            confidence = np.max(predictions)
                            confidence_percent = confidence * 100
                            
                            # Store history
                            st.session_state.prediction_history.append({
                                'class': selected_class,
                                'predicted': predicted_class,
                                'confidence': confidence,
                                'correct': selected_class == predicted_class,
                                'inference_time': inference_time
                            })
                            
                            # Display results in columns
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if selected_class == predicted_class:
                                    st.markdown(f"""
                                    <div class="prediction-high">
                                        <strong>✅ Correct!</strong>
                                        <br>Predicted: {predicted_class.capitalize()}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="prediction-low">
                                        <strong>❌ Incorrect</strong>
                                        <br>True: {selected_class.capitalize()}
                                        <br>Predicted: {predicted_class.capitalize()}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Confidence", f"{confidence_percent:.1f}%")
                            
                            with col3:
                                st.metric("Inference Time", f"{inference_time*1000:.1f} ms")
                            
                            # Confidence plot - FIXED: use width='stretch' instead of use_container_width
                            st.subheader("📊 Prediction Analysis")
                            fig_conf = plot_prediction_confidence(predictions)
                            st.plotly_chart(fig_conf, width='stretch')
                            
                            # Confidence entropy
                            fig_entropy, entropy_val, norm_entropy = plot_confidence_entropy(predictions)
                            st.plotly_chart(fig_entropy, width='stretch')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence Entropy", f"{entropy_val:.3f} bits")
                            with col2:
                                st.metric("Normalized Entropy", f"{norm_entropy:.2%}")
                            
                            # Top predictions
                            st.subheader("📋 Top Predictions")
                            top_k = st.slider("Number of predictions to show", 1, 10, 5)
                            top_indices = np.argsort(predictions[0])[::-1][:top_k]
                            top_classes = [CLASS_NAMES[i] for i in top_indices]
                            top_confidences = [predictions[0][i] for i in top_indices]
                            
                            df_top = pd.DataFrame({
                                'Class': [c.capitalize() for c in top_classes],
                                'Confidence': [f"{c*100:.2f}%" for c in top_confidences]
                            })
                            st.dataframe(df_top, use_container_width=True)
                            
                            # Confusion Matrix (if enabled and we have history)
                            if show_confusion and len(st.session_state.prediction_history) > 1:
                                st.subheader("📊 Confusion Matrix")
                                history_df = pd.DataFrame(st.session_state.prediction_history)
                                y_true = [CLASS_INDICES.get(cls, 0) for cls in history_df['class']]
                                y_pred = [CLASS_INDICES.get(pred, 0) for pred in history_df['predicted']]
                                
                                fig_cm = create_confusion_matrix_plot(y_true, y_pred)
                                if fig_cm:
                                    st.plotly_chart(fig_cm, width='stretch')
                        
            else:
                st.error(f"Failed to load image. Status code: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Research Metrics
    if st.session_state.prediction_history:
        st.subheader("📈 Research Metrics")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            correct = sum(history_df['correct'])
            accuracy = (correct / len(history_df)) * 100 if len(history_df) > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col2:
            avg_conf = history_df['confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with col3:
            avg_time = history_df['inference_time'].mean() * 1000
            st.metric("Avg Inference", f"{avg_time:.1f} ms")
        with col4:
            st.metric("Total Predictions", len(history_df))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧠 Advanced CIFAR-10 Research Classifier v2.0</p>
        <p style='font-size: 0.8rem;'>For research and educational purposes | Author: Md Abu Sufian</p>
        <p style='font-size: 0.8rem;'>Powered by TensorFlow, Streamlit, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
