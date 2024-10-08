import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

st.title("Satellite Image Classification")

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('best_model.h5')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model loading fails

# Mapping dictionary for class indices
map_dict = {0: 'Cloudy',
            1: 'Desert',
            2: 'Green Area',
            3: 'Water'}

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        if opencv_image is None:
            st.error("Error decoding image. Please upload a valid image file.")
            st.stop()

        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))

        # Display the uploaded image
        st.image(opencv_image, channels="RGB")

        # Preprocess the image for MobileNetV2
        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = np.expand_dims(resized, axis=0)  # Ensure correct shape for model input

        # Button to generate prediction
        if st.button("Generate Prediction"):
            try:
                # Predict and log output
                prediction = model.predict(img_reshape)
                predicted_class = np.argmax(prediction)

                # Log prediction output
                st.write(f"Prediction array: {prediction}")
                st.title(f"Predicted Label for the image is {map_dict[predicted_class]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
