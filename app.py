import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Define class names and base URL for example images
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main"

# Display the app header and description
st.image("Real_DL_architect.gif", use_column_width=True)
st.title("3D Natural Image Classification App")
st.write("This app demonstrates image classification into different classes using a web application.")
st.markdown("<span style='color:blue'>Author: Md Abu Sufian</span>", unsafe_allow_html=True)
st.write("......Visualization of Design and Coding Under Construction.........")

# Function to download and cache the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/blob/main/cifar10_cnn.h5?raw=true'
    model_path = 'cifar10_cnn.h5'
    if not os.path.isfile(model_path):
        with st.spinner(f'Downloading model...'):
            r = requests.get(model_url, stream=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(r.content)
            else:
                raise Exception("Error downloading model: ", r.status_code)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Sidebar for image upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to get the example image path from class name
def get_example_image_path(class_name):
    return f"{BASE_IMAGE_URL}/cifar_image_{class_name}_1.png"

# Process uploaded image and make predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_array = np.array(image) / 255.0
    image_array = image_array[np.newaxis, ...]
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    st.write(f"Model Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")

# Class selection and example image display
class_selection = st.selectbox("Or select a class to view an example:", class_names)
if class_selection:
    example_image_path = get_example_image_path(class_selection.lower())
    example_image = Image.open(requests.get(example_image_path, stream=True).raw)
    st.image(example_image, caption=f"Image Class of {class_selection}", use_column_width=True)

    # Optionally: Predict and display for the example image
    # Note: This part assumes you want to predict the example image automatically
    # Resize and preprocess the example image
    example_image_array = np.array(example_image.resize((32, 32))) / 255.0
    example_image_array = example_image_array[np.newaxis, ...]
    example_predictions = model.predict(example_image_array)
    example_predicted_class = class_names[np.argmax(example_predictions)]
    example_confidence = np.max(example_predictions)
    st.write(f"Image Class Prediction: {example_predicted_class}")
    st.write(f"Prediction Confidence: {example_confidence:.2%}")
