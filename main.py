import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

st.title("Satellite Image Classification")

# Load the pre-trained model
model = tf.keras.models.load_model('best_model.h5')

# Mapping dictionary for class indices
map_dict = {0: 'Cloudy',
            1: 'Desert',
            2: 'Green Area',
            3: 'Water'}

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))

    # Display the uploaded image
    st.image(opencv_image, channels="RGB")

    # Preprocess the image for MobileNetV2
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = np.expand_dims(resized, axis=0)  # Ensure correct shape for model input

    # Button to generate prediction
    generate_pred = st.button("Generate Prediction")
    if generate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title(f"Predicted Label for the image is {map_dict[prediction]}")
