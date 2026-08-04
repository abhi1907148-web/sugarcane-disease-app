import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.title("Sugarcane Leaf Disease Detector")
st.write("Upload a sugarcane leaf image to detect disease")

current_dir = os.path.dirname(os.path.abspath(__file__))
st.write("Files found in app folder:", os.listdir(current_dir))

model_path = os.path.join(current_dir, "sugarcane_mobilenetv2.tflite")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")dth=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
