import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Class names in CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

# Base URL for example images
BASE_IMAGE_URL = "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main"

# Display header image
st.image("Real_DL_architect.gif", use_column_width=True)

# Title and description
st.title("3D Natural Image Classification App")
st.write("This app demonstrates Image classification into different classes using web application.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.write("......Visualisation of Design and Coding Under Construction.........")

# Function to download and cache the model
@st.cache(allow_output_mutation=True)
def download_model(url, model_name):
    if not os.path.isfile(model_name):
        with st.spinner(f'Downloading {model_name}...'):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(model_name, 'wb') as f:
                    f.write(r.content)
            else:
                raise Exception("Error downloading model: ", r.status_code)
    return model_name

# Function to load and cache the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/blob/main/cifar10_cnn.h5?raw=true'
    model_path = download_model(model_url, 'cifar10_cnn.h5')
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Sidebar for image upload
st.sidebar.title("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to get the example image path from class name
def get_example_image_path(class_name):
    return f"{BASE_IMAGE_URL}/cifar_image_{class_name}_1.png"

# Main app logic
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)

  # Main app logic
def app():
    st.title("CIFAR-10 Image Classifier")
    model = load_model()  # Load your pre-trained model
  
# Dropdown for class selection
    class_selection = st.selectbox("Select a class to see an example image and prediction:", class_names)

# Make a prediction on the example image
    example_image = example_image.resize((32, 32))  # Resize to match model expected input

    # Predict and display the result
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array[np.newaxis, ...]  # Add a batch dimension
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)  # Get the highest probability value
    
    st.write(f"Model Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")  # Display as a percentage
  
class_selection = st.selectbox("Select a class to see an example image and prediction:", class_names)

if class_selection:
    example_image_path = get_example_image_path(class_selection.lower())
    st.write(f"Example image for class: {class_selection}")
    st.write(f"Example Image Prediction: {predicted_class}")
    example_image = Image.open(example_image_path)
    st.image(example_image_path, caption=f"{class_selection.capitalize()} Example", use_column_width=True)
    st.write(f"Prediction Confidence: {confidence:.2%}")  # Display as a percentage

if uploaded_file is not None:
        if class_selection == predicted_class:
            st.success(f"The model's prediction matches your selection: {predicted_class} with confidence {confidence:.2%}")
        else:
            st.error(f"The model's prediction does not match your selection. Predicted: {predicted_class} with confidence {confidence:.2%}")
