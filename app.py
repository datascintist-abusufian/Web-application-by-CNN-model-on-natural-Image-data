import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Assuming TensorFlow is already installed and you've checked its version
# print(tf.__version__)

# Class names in CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

@st.cache(allow_output_mutation=True)
def load_model():
    # Ensure the model path is correct and the model is properly saved
    return tf.keras.models.load_model('cifar10_cnn.h5')

model = load_model()
st.title("CIFAR-10 Image Classifier")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = np.array(image) / 255.0  # Normalize the image
    image = image[np.newaxis, ...]  # Add a batch dimension

    # User selection for class
    user_selected_class = st.selectbox("What do you think this image is?", class_names)
    st.write(f"You selected: {user_selected_class}")

    # Model prediction
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Model Prediction: {predicted_class}")

    # Compare the prediction with user selection
    if user_selected_class == predicted_class:
        st.success("The model's prediction matches your selection!")
    else:
        st.error("The model's prediction does not match your selection.")
