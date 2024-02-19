import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import random

# Define class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Base URL for the images
base_image_url = 'https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/'

# Function to download and cache the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'cifar10_cnn.h5'
    model_url = f"https://github.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/raw/main/{model_path}"
    if not os.path.isfile(model_path):
        with st.spinner('Downloading model...'):
            r = requests.get(model_url, allow_redirects=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(r.content)
            else:
                raise Exception('Failed to download model')
    model = load_model(model_path)
    return model

# Try loading the model, if this fails, the app will not proceed further
try:
    model = load_model()
except Exception as e:
    st.error("The app won't function without the model, please fix the error above.")
    st.stop()

# Sidebar for class selection
class_selection = st.sidebar.selectbox("Select a class:", class_names)

# When a class is selected, display a random image from that class and its prediction
if class_selection:
    # Generate a random image number. Adjust the range if you have a different number of images per class.
    image_number = random.randint(1, 10)  # Assuming there are 10 images per class
    image_url = f"{base_image_url}cifar_image_{class_selection.lower()}_{image_number}.png"

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Display the image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            st.image(image, caption=f"Random {class_selection.capitalize()} Image", use_column_width=True)

            # Preprocess the image and make a prediction
            image_array = np.array(image.resize((32, 32))) / 255.0
            image_array = image_array[np.newaxis, ...]  # Add batch dimension
            predictions = model.predict(image_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Display the prediction and confidence
            st.write(f"Predicted Class: {predicted_class.capitalize()}")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.error("Failed to fetch the example image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
