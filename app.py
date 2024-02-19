import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Define class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# List of raw image URLs
image_urls = [
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_airplane_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_automobile_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_bird_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_cat_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_deer_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_dog_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_frog_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_horse_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_ship_1.png",
    "https://raw.githubusercontent.com/datascintist-abusufian/Web-application-by-CNN-model-on-natural-Image-data/main/cifar_image_truck_1.png",
]
st.title('CIFAR-10 Image Classes')

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
        image_array = image_array[np.newaxis, ...]  # Add batch dimension
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        st.write(f"Model Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")
    except Exception as e:
        st.error(f"Failed to process the uploaded image: {e}")

# Display each example image with its class name
for idx, url in enumerate(image_urls):
    class_name = class_names[idx]  # Use the class names list
    st.subheader(f'Class: {class_name.capitalize()}')
    st.image(url, caption=f'{class_name.capitalize()} Image', use_column_width=True)

# The following section seems to be intended for selecting a class and displaying
# an example image from the preloaded set, but since we are already displaying all images,
# it's commented out. You can uncomment and adapt it if needed.
# 
# class_selection = st.selectbox("Or select a class to see an example image:", class_names)
# if class_selection:
#     # Logic to display selected class example image
#     # ...
