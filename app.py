import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Assuming TensorFlow is already installed and you've checked its version
# print(tf.__version__)

# Class names in CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

st.image("Real_DL_architect.gif", use_column_width=True)

# Title and description
st.title("3D Natural Image Classification App")
st.write("This app demonstrates Image classification into different classes using web application.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.markdown(
st.write( " Design and Coding Under Construction")
@st.cache(allow_output_mutation=True)
def download_model(url, model_name):
    """
    Download the model from a given URL if it's not already in the cache.
    """
    if not os.path.isfile(model_name):
        with st.spinner(f'Downloading {model_name}...'):
            r = requests.get(url)
            with open(model_name, 'wb') as f:
                f.write(r.content)
    return model_name

@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/blob/main/cifar10_cnn.h5?raw=true'
    model_path = download_model(model_url, 'cifar10_cnn.h5')
    return tf.keras.models.load_model(model_path)

model = load_model()
st.title("CIFAR-10 Image Classifier")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array[np.newaxis, ...]  # Add a batch dimension

    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)  # Get the highest probability value
    
    # Display the prediction and confidence on the main page
    st.write(f"Model Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")  # Display as a percentage

# Allow the user to select a class to filter predictions
class_selection = st.selectbox("Select a class to filter predictions:", class_names)

if uploaded_file is not None and class_selection:
    st.write(f"You selected: {class_selection}")
    if class_selection == predicted_class:
        st.success(f"The model's prediction matches your selection: {predicted_class} with confidence {confidence:.2%}")
    else:
        st.error(f"The model's prediction does not match your selection. Predicted: {predicted_class} with confidence {confidence:.2%}")
