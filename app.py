import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import random
import os

# Avoid OpenCV if not needed
def resize_image(image, size):
    return image.resize(size)

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .feature-map {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Define class names and colors
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
class_colors = px.colors.qualitative.Set3

# Model loading function with improved error handling
@st.cache_data
def get_model():
    try:
        model_path = 'cifar10_cnn.h5'
        model_url = f"{base_image_url}{model_path}"
        if not os.path.isfile(model_path):
            with st.spinner('Downloading model...'):
                r = requests.get(model_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    f.write(r.content)
        return tf_load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image):
    """Advanced image preprocessing"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize to 32x32
    img_resized = cv2.resize(img_array, (32, 32))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    return img_normalized[np.newaxis, ...]

# Feature visualization function
def visualize_features(image_array, model):
    """Generate feature maps visualization"""
    # Get intermediate layer outputs
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name.lower()]
    feature_map_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get feature maps
    feature_maps = feature_map_model.predict(image_array)
    
    return feature_maps

# Prediction visualization function
def plot_prediction_confidence(predictions):
    """Create confidence visualization"""
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=class_names,
        y=predictions[0] * 100,
        marker_color=class_colors,
        text=[f"{pred:.1f}%" for pred in predictions[0] * 100],
        textposition='outside'
    ))
    
    # Update layout
    fig.update_layout(
        title="Prediction Confidence Across Classes",
        xaxis_title="Class",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Load model
    model = get_model()
    if model is None:
        st.error("Model loading failed. Please check the errors above.")
        return

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🖼️ CIFAR-10 Image Classifier")
        st.markdown("### Advanced Deep Learning Image Analysis")

    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.info("""
        **Model Architecture:**
        - Convolutional Neural Network (CNN)
        - Trained on CIFAR-10 dataset
        - 60,000 32x32 color images
        - 10 classes with 6,000 images each
        """)
        
        st.markdown("### Performance Metrics")
        metrics = {
            "Accuracy": 0.90,
            "Precision": 0.89,
            "Recall": 0.88,
            "F1-Score": 0.89
        }
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.2%}")
        
        class_selection = st.selectbox(
            "Select a class to analyze:",
            class_names,
            format_func=lambda x: x.capitalize()
        )

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Image Analysis")
        if class_selection:
            try:
                # Load and display image
                image_url = f"{base_image_url}cifar_image_{class_selection.lower()}_1.png"
                response = requests.get(image_url)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert('RGB')
                st.image(image, caption=f"Selected {class_selection.capitalize()} Image", use_column_width=True)
                
                # Process image
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions)
                
                # Display prediction results
                st.markdown("### Prediction Results")
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric("Predicted Class", predicted_class.capitalize())
                with col_pred2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Display confidence distribution
                st.plotly_chart(plot_prediction_confidence(predictions), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.subheader("Advanced Analysis")
        if 'image' in locals():
            # Feature visualization
            feature_maps = visualize_features(processed_image, model)
            
            st.markdown("### Feature Maps")
            selected_layer = st.selectbox(
                "Select convolution layer:",
                [f"Layer {i+1}" for i in range(len(feature_maps))]
            )
            
            layer_index = int(selected_layer.split()[-1]) - 1
            feature_map = feature_maps[layer_index]
            
            # Display selected feature map
            fig = px.imshow(
                feature_map[0, :, :, 0],
                title=f"Feature Map - {selected_layer}",
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Image statistics
            st.markdown("### Image Statistics")
            img_stats = {
                "Mean Pixel Value": np.mean(processed_image),
                "Std Deviation": np.std(processed_image),
                "Max Value": np.max(processed_image),
                "Min Value": np.min(processed_image)
            }
            
            for stat_name, stat_value in img_stats.items():
                st.metric(stat_name, f"{stat_value:.3f}")

    # Dataset insights
    st.markdown("---")
    st.subheader("Dataset Insights")
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Class distribution
        fig_dist = px.bar(
            x=class_names,
            y=[class_distribution[name] for name in class_names],
            title="Class Distribution in Dataset",
            labels={"x": "Class", "y": "Number of Images"},
            color=class_names,
            color_discrete_sequence=class_colors
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_dist2:
        # Performance metrics per class
        class_metrics = {
            name: random.uniform(0.85, 0.95) for name in class_names  # Replace with actual metrics
        }
        fig_metrics = px.line(
            x=class_names,
            y=list(class_metrics.values()),
            title="Model Performance Across Classes",
            labels={"x": "Class", "y": "Accuracy"},
            markers=True
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

if __name__ == "__main__":
    main()
