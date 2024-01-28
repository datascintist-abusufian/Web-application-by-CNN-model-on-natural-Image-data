import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Class names in CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('cifar10_cnn.h5')

model = load_model()
st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.array(image) / 255.0
    image = image[np.newaxis, ...]
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")
