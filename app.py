import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import random
import os

# Define class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Base URL for the images
base_image_url = 'https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/'

# Function to download and cache the TensorFlow model
@st.cache(allow_output_mutation=True)
def get_model():
    model_path = 'cifar10_cnn.h5'
    model_url = f"{base_image_url}{model_path}"
    if not os.path.isfile(model_path):
        with st.spinner('Downloading model...'):
            r = requests.get(model_url, stream=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(r.content)
            else:
                st.error('Failed to download model: HTTP status code {}'.format(r.status_code))
                return None
    return load_model(model_path)

# Try loading the model, if this fails, the app will not proceed further
model = get_model()
if model is None:
    st.error("The app won't function without the model, please fix the error above.")
    st.stop()

# Create a select box for the user to select a class
class_selection = st.selectbox('Select a class', class_names)

# When a class is selected, display a random image from that class and its prediction
if class_selection:
    # Get a random image from the selected class
    image_path = f"{class_selection}/{random.choice(os.listdir(class_selection))}"
    image_url = f"{base_image_url}{image_path}"
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Display the image
    st.image(img, caption=f"An example of {class_selection}", use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 32, 32, 3))

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display the prediction
    st.write(f"The model predicts this image is a {predicted_class}")
