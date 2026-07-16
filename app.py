import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
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
IMAGE_SIZE = 224  # For transfer learning models

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
def preprocess_image(image, model_type='CNN', target_size=(32, 32)):
    """Preprocess image based on model type"""
    try:
        # Resize based on model type
        if model_type in ['ResNet50', 'VGG16', 'MobileNetV2']:
            img_resized = image.resize((224, 224))
        else:
            img_resized = image.resize((32, 32))
        
        # Convert to array
        img_array = np.array(img_resized)
        
        # Normalize based on model type
        if model_type == 'CNN':
            img_array = img_array / 255.0
        elif model_type == 'ResNet50':
            img_array = resnet_preprocess(img_array.astype(np.float32))
        elif model_type == 'VGG16':
            img_array = vgg_preprocess(img_array.astype(np.float32))
        elif model_type == 'MobileNetV2':
            img_array = mobilenet_preprocess(img_array.astype(np.float32))
        
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# ============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ============================================================================
def plot_prediction_confidence(predictions):
    """Create enhanced prediction confidence visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Prediction Confidence by Class', 'Confidence Distribution'],
        row_heights=[0.7, 0.3]
    )
    
    # Bar chart
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
    
    # Confidence distribution
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
    
    # Confidence distribution
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
    
    # Dimensionality reduction
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
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title='Confusion Matrix'
    )
    
    fig.update_layout(height=500)
    return fig

# ============================================================================
# IMAGE ANALYSIS FUNCTIONS
# ============================================================================
def analyze_image_quality(image):
    """Analyze image quality metrics"""
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate metrics
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()
    
    # Color analysis
    if len(img_array.shape) == 3:
        color_mean = img_array.mean(axis=(0, 1))
        color_std = img_array.std(axis=(0, 1))
        color_hist = np.array([
            np.histogram(img_array[:,:,i], bins=10, range=(0, 255))[0]
            for i in range(3)
        ])
    else:
        color_mean = None
        color_std = None
        color_hist = None
    
    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'brightness': brightness,
        'color_mean': color_mean,
        'color_std': color_std,
        'color_hist': color_hist,
        'shape': img_array.shape,
        'size': image.size
    }

def generate_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM visualization"""
    try:
        # This is a simplified Grad-CAM implementation
        # For full implementation, you'd need to modify the model architecture
        last_conv_layer = model.layers[-4]  # Approximate last conv layer
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv output
        heatmap = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(heatmap, pooled_grads), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        
        return heatmap.numpy()
        
    except:
        return None

# ============================================================================
# BATCH ANALYSIS FUNCTION
# ============================================================================
def analyze_batch_images(df, model, model_type):
    """Analyze multiple images from a dataframe"""
    results = []
    
    for idx, row in df.iterrows():
        try:
            # Load image
            img = Image.open(BytesIO(requests.get(row['image_url']).content)).convert('RGB')
            
            # Preprocess
            processed = preprocess_image(img, model_type)
            
            # Predict
            predictions = model.predict(processed, verbose=0)
            pred_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)
            
            results.append({
                'image_id': idx,
                'true_class': row.get('true_class', 'Unknown'),
                'predicted_class': pred_class,
                'confidence': confidence,
                'correct': row.get('true_class', '') == pred_class if 'true_class' in row else None
            })
            
        except Exception as e:
            results.append({
                'image_id': idx,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
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
        batch_mode = st.checkbox("Batch Analysis Mode", value=False)
        
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
    
    # Main content
    if not batch_mode:
        # Single image analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Image Analysis",
            "📊 Predictions",
            "🎨 Feature Visualization",
            "📈 Research Metrics",
            "📝 Report"
        ])
        
        with tab1:
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
                
                # Sample images
                try:
                    image_url = f"{BASE_IMAGE_URL}cifar_image_{selected_class.lower()}_1.png"
                    response = requests.get(image_url)
                    
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        
                        # Display image
                        st.image(image, caption=f"Sample {selected_class.capitalize()} Image", use_container_width=True)
                        
                        # Upload option
                        uploaded_file = st.file_uploader(
                            "Or upload your own image",
                            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
                        )
                        
                        if uploaded_file:
                            image = Image.open(uploaded_file).convert('RGB')
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                        
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
                                    
                                    # Image quality analysis
                                    st.subheader("📷 Image Quality Metrics")
                                    quality_metrics = analyze_image_quality(image)
                                    
                                    q_col1, q_col2, q_col3 = st.columns(3)
                                    with q_col1:
                                        st.metric("Sharpness", f"{quality_metrics['sharpness']:.1f}")
                                    with q_col2:
                                        st.metric("Contrast", f"{quality_metrics['contrast']:.1f}")
                                    with q_col3:
                                        st.metric("Brightness", f"{quality_metrics['brightness']:.1f}")
                                    
                                    # Grad-CAM
                                    if show_gradcam:
                                        st.subheader("🔍 Grad-CAM Visualization")
                                        heatmap = generate_gradcam(st.session_state.model, processed, predicted_idx)
                                        if heatmap is not None:
                                            fig_heatmap = go.Figure()
                                            fig_heatmap.add_trace(go.Heatmap(
                                                z=heatmap,
                                                colorscale='Viridis',
                                                showscale=True
                                            ))
                                            fig_heatmap.update_layout(height=300)
                                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                    else:
                        st.error(f"Failed to load image. Status code: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with tab2:
            if st.session_state.prediction_history:
                st.subheader("📊 Prediction Analysis")
                
                # Confidence plot
                fig_conf = plot_prediction_confidence(predictions)
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Confidence entropy
                fig_entropy, entropy_val, norm_entropy = plot_confidence_entropy(predictions)
                st.plotly_chart(fig_entropy, use_container_width=True)
                
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
                
                # Classification report
                if show_confusion:
                    st.subheader("📊 Classification Report")
                    
                    # Mock data for demonstration
                    y_true = [CLASS_INDICES[cls] for cls in CLASS_NAMES]
                    y_pred = [CLASS_INDICES[predicted_class]] * len(CLASS_NAMES)
                    
                    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(df_report.round(4), use_container_width=True)
                    
                    # Confusion matrix
                    fig_cm = create_confusion_matrix_plot(y_true, y_pred)
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("ℹ️ Perform an analysis first to see predictions here.")
        
        with tab3:
            st.subheader("🎨 Feature Space Visualization")
            
            if show_feature_space:
                # Generate sample features (mock data for demonstration)
                np.random.seed(42)
                n_samples = 100
                features = np.random.randn(n_samples, 128)
                labels = np.random.randint(0, 10, n_samples)
                
                # PCA visualization
                st.markdown("#### PCA Visualization")
                fig_pca = plot_feature_space(features, labels, 'PCA')
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                # t-SNE visualization
                st.markdown("#### t-SNE Visualization")
                fig_tsne = plot_feature_space(features, labels, 't-SNE')
                if fig_tsne:
                    st.plotly_chart(fig_tsne, use_container_width=True)
                
                # Cluster analysis
                st.markdown("#### Cluster Analysis")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42)
                clusters = kmeans.fit_predict(features)
                
                fig_cluster = px.scatter(
                    x=features[:, 0],
                    y=features[:, 1],
                    color=clusters.astype(str),
                    title='K-Means Clustering of Features',
                    labels={'x': 'Feature 1', 'y': 'Feature 2'},
                    color_discrete_sequence=COLORS
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.info("Feature space visualization is disabled. Enable it in the sidebar.")
        
        with tab4:
            st.subheader("📈 Research Metrics")
            
            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                
                # Performance metrics
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
                
                # Confusion matrix
                st.subheader("Confusion Matrix Analysis")
                y_true = [CLASS_INDICES.get(cls, 0) for cls in history_df['class']]
                y_pred = [CLASS_INDICES.get(pred, 0) for pred in history_df['predicted']]
                
                fig_cm = create_confusion_matrix_plot(y_true, y_pred)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Performance over time
                st.subheader("Performance Over Time")
                history_df['timestamp'] = pd.to_datetime(history_df.index, unit='s')
                history_df['correct_numeric'] = history_df['correct'].astype(int)
                
                fig_perf = px.line(
                    history_df,
                    x='timestamp',
                    y='confidence',
                    title='Confidence Over Time',
                    labels={'confidence': 'Confidence', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.info("Perform analyses to see research metrics here.")
        
        with tab5:
            st.subheader("📝 Research Report")
            
            if st.session_state.prediction_history:
                # Generate report
                history_df = pd.DataFrame(st.session_state.prediction_history)
                
                st.markdown("### 📊 Summary Statistics")
                summary = {
                    'Total Predictions': len(history_df),
                    'Correct Predictions': sum(history_df['correct']),
                    'Accuracy': f"{(sum(history_df['correct'])/len(history_df))*100:.2f}%",
                    'Average Confidence': f"{history_df['confidence'].mean()*100:.2f}%",
                    'Average Inference Time': f"{history_df['inference_time'].mean()*1000:.2f} ms",
                    'Model': st.session_state.model_type
                }
                
                for key, value in summary.items():
                    st.write(f"**{key}:** {value}")
                
                # Class-wise performance
                st.markdown("### 📈 Class-wise Performance")
                class_performance = []
                for class_name in CLASS_NAMES:
                    class_data = history_df[history_df['class'] == class_name]
                    if len(class_data) > 0:
                        class_performance.append({
                            'Class': class_name.capitalize(),
                            'Count': len(class_data),
                            'Accuracy': f"{(sum(class_data['correct'])/len(class_data))*100:.1f}%",
                            'Avg Confidence': f"{class_data['confidence'].mean()*100:.1f}%"
                        })
                
                df_perf = pd.DataFrame(class_performance)
                st.dataframe(df_perf, use_container_width=True)
                
                # Export report
                st.markdown("### 📥 Export Report")
                if st.button("Generate Full Report", use_container_width=True):
                    report_data = {
                        'metadata': {
                            'model': st.session_state.model_type,
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'total_predictions': len(history_df)
                        },
                        'performance': summary,
                        'class_performance': class_performance,
                        'history': history_df.to_dict('records')
                    }
                    
                    # Convert to JSON
                    import json
                    json_str = json.dumps(report_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=json_str,
                        file_name=f"research_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.info("Perform analyses to generate a research report.")
    
    else:
        # Batch analysis mode
        st.subheader("📊 Batch Analysis Mode")
        
        st.info("""
        ### Batch Analysis
        Upload a CSV file with image URLs for batch processing.
        The CSV should contain a column named 'image_url' with image URLs.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing batch images..."):
                        results = analyze_batch_images(df, st.session_state.model, st.session_state.model_type)
                        
                        # Display results
                        st.subheader("Batch Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Summary statistics
                        if 'correct' in results.columns:
                            valid_results = results[results['correct'].notna()]
                            if len(valid_results) > 0:
                                accuracy = sum(valid_results['correct']) / len(valid_results)
                                st.metric("Batch Accuracy", f"{accuracy*100:.2f}%")
                                
                                # Confusion matrix
                                y_true = [CLASS_INDICES.get(cls, 0) for cls in valid_results['true_class']]
                                y_pred = [CLASS_INDICES.get(pred, 0) for pred in valid_results['predicted_class']]
                                
                                fig_cm = create_confusion_matrix_plot(y_true, y_pred)
                                st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Batch Results",
                            data=csv,
                            file_name=f"batch_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
    
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
