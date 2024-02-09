import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
from PIL import Image, UnidentifiedImageError

# Define class names and base URL for example images
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main"

# Display the app header and description
st.image("Real_DL_architect.gif", use_column_width=True)
st.title("3D Natural Image Classification App")
st.write("This app demonstrates image classification into different classes using a web application.")
st.markdown("<span style='color:blue'>Author: Md Abu Sufian</span>", unsafe_allow_html=True)

# Function to download and cache the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/cifar10_cnn.h5'
    model_path = 'cifar10_cnn.h5'
    if not os.path.isfile(model_path):
        with st.spinner('Downloading model...'):
            try:
                r = requests.get(model_url, stream=True)
                if r.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(r.content)
                else:
                    raise Exception("Error downloading model: ", r.status_code)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                raise e
    return tf.keras.models.load_model(model_path)

# Try loading the model, if this fails, the app will not proceed further
try:
    model = load_model()
except Exception as e:
    st.error("The app won't function without the model, please fix the error above.")
    st.stop()

# Sidebar for image upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Process uploaded image and make predictions
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB').resize((32, 32))
        st.image(image, caption='Uploaded Image', width=300)
        image_array = np.array(image) / 255.0
        image_array = image_array[np.newaxis, ...]
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        st.write(f"Model Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")
    except Exception as e:
        st.error(f"Failed to process the uploaded image: {e}")

# Class selection and example image display
class_selection = st.selectbox("Upload Or select a class to get prediction accuracy:", class_names)
if class_selection:
    try:
        example_image_filename = f"cifar_image_{class_selection.lower()}_1.png"
        example_image_url = f"{BASE_IMAGE_URL}/{example_image_filename}"
        response = requests.get(example_image_url, stream=True) # Define response here
try:
    response = requests.get(example_image_url, stream=True)
    if response.status_code == 200:
        from io import BytesIO
        example_image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(example_image, caption=f"Image Class of {class_selection}", use_column_width=True)
        
        example_image_array = np.array(example_image.resize((32, 32))) / 255.0
        example_image_array = example_image_array[np.newaxis, ...]  # Add batch dimension
        example_predictions = model.predict(example_image_array)
        example_predicted_class = class_names[np.argmax(example_predictions)]
        example_confidence = np.max(example_predictions)
        st.write(f"Image Class Prediction: {example_predicted_class}")
        st.write(f"Prediction Confidence: {example_confidence:.2%}")
    else:
        st.error("Failed to fetch example image. The image may not exist at the provided URL.")
except Exception as e:
    st.error(f"An error occurred: {e}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")


