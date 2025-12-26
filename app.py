import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

st.title("Face Mask Detection System")
st.write("Detects whether a person is wearing a face mask using a CNN model.")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_mobilenetv2.keras")

model = load_model()

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image_bgr):
    image = cv2.resize(image_bgr, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# App settings
# --------------------------------------------------
THRESHOLD = 0.7

mode = st.radio(
    "Select Input Mode",
    ("Upload Image", "Webcam Snapshot")
)

# ==================================================
# IMAGE UPLOAD MODE
# ==================================================
if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch")

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed = preprocess_image(image_bgr)
        prediction = model.predict(processed)

        without_mask_prob = prediction[0][0]
        with_mask_prob = prediction[0][1]

        if with_mask_prob >= THRESHOLD:
            st.success(
                f"Prediction: With Mask\n\n"
                f"Confidence: {with_mask_prob:.4f}"
            )
        else:
            st.error(
                f"Prediction: Without Mask\n\n"
                f"Confidence: {without_mask_prob:.4f}"
            )

        st.subheader("Confidence Scores")
        st.write(f"With Mask: {with_mask_prob:.4f}")
        st.write(f"Without Mask: {without_mask_prob:.4f}")

# ==================================================
# WEBCAM SNAPSHOT MODE
# ==================================================
elif mode == "Webcam Snapshot":

    webcam_image = st.camera_input("Capture image")

    if webcam_image is not None:
        image = Image.open(webcam_image).convert("RGB")
        st.image(image, width="stretch")

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed = preprocess_image(image_bgr)
        prediction = model.predict(processed)

        without_mask_prob = prediction[0][0]
        with_mask_prob = prediction[0][1]

        if with_mask_prob >= THRESHOLD:
            st.success(
                f"Prediction: With Mask\n\n"
                f"Confidence: {with_mask_prob:.4f}"
            )
        else:
            st.error(
                f"Prediction: Without Mask\n\n"
                f"Confidence: {without_mask_prob:.4f}"
            )

        st.subheader("Confidence Scores")
        st.write(f"With Mask: {with_mask_prob:.4f}")
        st.write(f"Without Mask: {without_mask_prob:.4f}")
