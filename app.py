import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stAlert {margin-top: 1rem;}
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

# Constants
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MODEL_URL = "https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_cnn.h5"
BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/"

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model_from_url():
    """Load the model from URL or local file"""
    try:
        model_path = 'cifar10_cnn.h5'
        if not os.path.exists(model_path):
            with st.spinner('Downloading model... This may take a few minutes...'):
                response = requests.get(MODEL_URL, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success("Model downloaded successfully!")
                else:
                    st.error(f"Failed to download model. Status code: {response.status_code}")
                    return None
        
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image
        img_resized = image.resize((32, 32))
        
        # Convert to array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def plot_predictions(predictions):
    """Create prediction confidence visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=CLASS_NAMES,
        y=predictions[0] * 100,
        marker_color=px.colors.qualitative.Set3,
        text=[f"{pred:.1f}%" for pred in predictions[0] * 100],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Prediction Confidence by Class",
        xaxis_title="Class",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.markdown("### Deep Learning Image Classification")
    
    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.info("""
        **About the Model:**
        - CNN Architecture
        - CIFAR-10 Dataset
        - 60,000 Training Images
        - 10 Classes
        """)
        
        # Class selection
        selected_class = st.selectbox(
            "Select image class:",
            CLASS_NAMES,
            format_func=lambda x: x.capitalize()
        )
    
    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model_from_url()
    
    if st.session_state.model is None:
        st.error("Failed to load model. Please check errors above.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            # Load and display image
            image_url = f"{BASE_IMAGE_URL}cifar_image_{selected_class.lower()}_1.png"
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                st.image(image, caption=f"{selected_class.capitalize()} Image", use_column_width=True)
                
                # Process image and predict
                with st.spinner("Analyzing image..."):
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        predictions = st.session_state.model.predict(processed_image)
                        predicted_class = CLASS_NAMES[np.argmax(predictions)]
                        confidence = np.max(predictions)
                        
                        # Display results
                        st.markdown("### Prediction Results")
                        col_pred1, col_pred2 = st.columns(2)
                        with col_pred1:
                            st.metric("Predicted Class", predicted_class.capitalize())
                        with col_pred2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Show confidence distribution
                        st.plotly_chart(plot_predictions(predictions), use_container_width=True)
            else:
                st.error(f"Failed to load image. Status code: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### Image Analysis")
        if 'image' in locals():
            # Image statistics
            st.markdown("#### Image Statistics")
            img_array = np.array(image)
            stats = {
                "Mean RGB": f"({img_array[:,:,0].mean():.1f}, {img_array[:,:,1].mean():.1f}, {img_array[:,:,2].mean():.1f})",
                "Standard Deviation": f"{img_array.std():.1f}",
                "Size": f"{image.size[0]}x{image.size[1]}",
                "Mode": image.mode
            }
            
            for name, value in stats.items():
                st.metric(name, value)
    
    # Dataset information
    st.markdown("---")
    st.subheader("Dataset Distribution")
    
    # Example distribution data
    distribution = {cls: 6000 for cls in CLASS_NAMES}  # CIFAR-10 has 6000 images per class
    
    fig = px.bar(
        x=list(distribution.keys()),
        y=list(distribution.values()),
        title="Images per Class in CIFAR-10",
        labels={"x": "Class", "y": "Number of Images"},
        color=list(distribution.keys()),
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
        ### Disclaimer
        This is a demonstration model and should not be used for critical applications. 
        The predictions are for educational purposes only.
    """)

if __name__ == "__main__":
    main()
