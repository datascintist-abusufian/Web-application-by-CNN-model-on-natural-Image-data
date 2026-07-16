import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
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

BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/"

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
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}

initialize_session_state()

# ============================================================================
# MODEL CREATION FUNCTIONS
# ============================================================================
def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Create a simple CNN model for CIFAR-10"""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(base_model_class, input_shape=(32, 32, 3), num_classes=10):
    """Create a transfer learning model with proper input handling"""
    # Handle different input sizes for different base models
    if base_model_class == ResNet50:
        target_size = (224, 224)
    elif base_model_class == VGG16:
        target_size = (224, 224)
    elif base_model_class == MobileNetV2:
        target_size = (224, 224)
    else:
        target_size = (32, 32)
    
    # Create input layer with correct size
    inputs = Input(shape=target_size + (3,))
    
    # Load base model with correct input size
    base_model = base_model_class(
        weights='imagenet', 
        include_top=False, 
        input_tensor=inputs
    )
    base_model.trainable = False
    
    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, target_size

def get_model_info(model_type):
    """Get model information"""
    info = {
        'CNN': {
            'name': 'Custom CNN',
            'input_size': (32, 32, 3),
            'params': '~1.2M',
            'description': 'Custom CNN with batch normalization and dropout'
        },
        'ResNet50': {
            'name': 'ResNet50',
            'input_size': (224, 224, 3),
            'params': '~25.6M',
            'description': 'ResNet50 with transfer learning'
        },
        'VGG16': {
            'name': 'VGG16',
            'input_size': (224, 224, 3),
            'params': '~14.7M',
            'description': 'VGG16 with transfer learning'
        },
        'MobileNetV2': {
            'name': 'MobileNetV2',
            'input_size': (224, 224, 3),
            'params': '~3.5M',
            'description': 'MobileNetV2 with transfer learning'
        }
    }
    return info.get(model_type, info['CNN'])

# ============================================================================
# MODEL LOADING WITH FALLBACK
# ============================================================================
@st.cache_resource
def load_or_create_model(model_type='CNN'):
    """Load model from file or create a new one if not available"""
    try:
        model = None
        model_path = f'models/cifar10_{model_type.lower()}.h5'
        os.makedirs('models', exist_ok=True)
        
        # Try to load from local file
        if os.path.exists(model_path):
            try:
                with st.spinner(f"Loading {model_type} model from local file..."):
                    model = load_model(model_path)
                    st.success(f"✅ {model_type} model loaded from local file!")
                    return model, 'local'
            except Exception as e:
                st.warning(f"Could not load local model: {str(e)}")
        
        # Try to download from URL
        url = f"https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_{model_type.lower()}.h5"
        try:
            with st.spinner(f"Downloading {model_type} model from URL..."):
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    model = load_model(model_path)
                    st.success(f"✅ {model_type} model downloaded and loaded!")
                    return model, 'download'
        except Exception as e:
            st.info(f"Model not found online. Creating new {model_type} model...")
        
        # If no model found, create one
        with st.spinner(f"🔄 Creating new {model_type} model... This may take a moment..."):
            info = get_model_info(model_type)
            input_shape = info['input_size']
            
            if model_type == 'CNN':
                model = create_cnn_model(input_shape=input_shape)
                st.session_state.model_info['input_size'] = input_shape
            elif model_type in ['ResNet50', 'VGG16', 'MobileNetV2']:
                base_class = {
                    'ResNet50': ResNet50,
                    'VGG16': VGG16,
                    'MobileNetV2': MobileNetV2
                }[model_type]
                model, target_size = create_transfer_learning_model(base_class)
                st.session_state.model_info['input_size'] = (target_size, target_size, 3)
            else:
                model = create_cnn_model()
                st.session_state.model_info['input_size'] = (32, 32, 3)
            
            # Save the model
            model.save(model_path)
            st.success(f"✅ New {model_type} model created and saved!")
            return model, 'created'
        
    except Exception as e:
        st.error(f"Error loading/creating model: {str(e)}")
        return None, None

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image, model_type='CNN'):
    """Preprocess image based on model type"""
    try:
        info = get_model_info(model_type)
        target_size = info['input_size'][:2]  # Get (height, width)
        
        # Resize image
        img_resized = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img_resized)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
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
        
        # Show model info
        info = get_model_info(model_type)
        st.markdown(f"""
        <div class="info-box">
            <strong>📋 Model Info</strong><br>
            <strong>Name:</strong> {info['name']}<br>
            <strong>Input Size:</strong> {info['input_size'][0]}x{info['input_size'][1]}<br>
            <strong>Parameters:</strong> {info['params']}<br>
            <strong>Description:</strong> {info['description']}
        </div>
        """, unsafe_allow_html=True)
        
        # Load model button
        if st.button("🔄 Load/Create Model", use_container_width=True):
            with st.spinner(f"Loading/Creating {model_type} model..."):
                model, source = load_or_create_model(model_type)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_loaded = True
                    
                    if source == 'local':
                        st.success(f"✅ {model_type} model loaded from local file!")
                    elif source == 'download':
                        st.success(f"✅ {model_type} model downloaded and loaded!")
                    else:
                        st.success(f"✅ New {model_type} model created and loaded!")
                else:
                    st.error("❌ Failed to load/create model")
        
        # Check if model is loaded
        if st.session_state.model_loaded and st.session_state.model is not None:
            st.markdown("---")
            st.success(f"✅ {st.session_state.model_type} model is ready!")
            
            # Model performance stats
            if st.session_state.prediction_history:
                st.subheader("📊 Performance")
                st.metric("Total Predictions", len(st.session_state.prediction_history))
                correct = sum([1 for p in st.session_state.prediction_history if p.get('correct', False)])
                if len(st.session_state.prediction_history) > 0:
                    st.metric("Accuracy", f"{(correct/len(st.session_state.prediction_history))*100:.1f}%")
        else:
            st.warning("⚠️ No model loaded. Click 'Load/Create Model' to get started.")
    
    # Check model
    if not st.session_state.model_loaded or st.session_state.model is None:
        st.warning("⚠️ Please load a model from the sidebar to begin analysis.")
        st.info("💡 Click the 'Load/Create Model' button to load or create a model.")
        
        # Show sample images
        st.subheader("📸 Sample Images")
        cols = st.columns(5)
        for idx, class_name in enumerate(CLASS_NAMES[:5]):
            with cols[idx]:
                try:
                    image_url = f"{BASE_IMAGE_URL}cifar_image_{class_name.lower()}_1.png"
                    response = requests.get(image_url, timeout=5)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        st.image(image, caption=class_name.capitalize(), width=None)
                except:
                    pass
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
            response = requests.get(image_url, timeout=5)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                st.image(image, caption=f"Sample {selected_class.capitalize()} Image", width=None)
                
                # Upload option
                uploaded_file = st.file_uploader(
                    "Or upload your own image",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", width=None)
                
                if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        processed = preprocess_image(image, st.session_state.model_type)
                        
                        if processed is not None:
                            start_time = time.time()
                            predictions = st.session_state.model.predict(processed, verbose=0)
                            inference_time = time.time() - start_time
                            
                            predicted_idx = np.argmax(predictions)
                            predicted_class = CLASS_NAMES[predicted_idx]
                            confidence = np.max(predictions)
                            
                            st.session_state.prediction_history.append({
                                'class': selected_class,
                                'predicted': predicted_class,
                                'confidence': confidence,
                                'correct': selected_class == predicted_class,
                                'inference_time': inference_time
                            })
                            
                            # Display results
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
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            with col3:
                                st.metric("Inference Time", f"{inference_time*1000:.1f} ms")
                            
                            # Confidence plot
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
                            
                            # Confusion Matrix
                            if len(st.session_state.prediction_history) > 1:
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
        <p style='font-size: 0.8rem;'>For research and educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
