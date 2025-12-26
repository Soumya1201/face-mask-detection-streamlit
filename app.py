import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection with Grad-CAM",
    layout="centered"
)

st.title("Face Mask Detection with Explainability (Grad-CAM)")

# --------------------------------------------------
# Detect Streamlit Cloud
# --------------------------------------------------
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "false") == "true"

# --------------------------------------------------
# Load model
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
# Grad-CAM (CORRECT for YOUR MobileNetV2)
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model, class_index):

    last_conv_layer = model.get_layer("out_relu")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val < 1e-6:
        return None

    heatmap /= max_val
    return heatmap.numpy()

# --------------------------------------------------
# Grad-CAM metrics
# --------------------------------------------------
def gradcam_metrics(heatmap):
    total = np.sum(heatmap)
    if total == 0:
        raise ValueError("Empty heatmap")

    h = heatmap.shape[0]

    return {
        "Mean Activation": np.mean(heatmap),
        "Max Activation": np.max(heatmap),
        "High Activation Ratio (>0.6)": np.sum(heatmap > 0.6) / heatmap.size,
        "Upper Face Contribution": np.sum(heatmap[:h // 2, :]) / total,
        "Lower Face Contribution": np.sum(heatmap[h // 2:, :]) / total,
        "Red Region Contribution (High Evidence)": np.sum(heatmap >= 0.66) / heatmap.size,
        "Yellow Region Contribution (Moderate Evidence)": np.sum((heatmap >= 0.33) & (heatmap < 0.66)) / heatmap.size,
        "Blue Region Contribution (Low Evidence)": np.sum(heatmap < 0.33) / heatmap.size,
    }

# --------------------------------------------------
# Overlay Grad-CAM
# --------------------------------------------------
def overlay_gradcam(image_bgr, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

# --------------------------------------------------
# App settings
# --------------------------------------------------
THRESHOLD = 0.7

if IS_CLOUD:
    mode = st.radio("Select Input Mode", ("Upload Image", "Webcam Snapshot"))
    st.info("Live video is disabled on Streamlit Cloud.")
else:
    mode = st.radio("Select Input Mode", ("Upload Image", "Webcam Snapshot", "Live Video"))

# ==================================================
# IMAGE UPLOAD
# ==================================================
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch")

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed = preprocess_image(image_bgr)
        prediction = model.predict(processed)

        mask_prob = prediction[0][1]
        no_mask_prob = prediction[0][0]

        if mask_prob >= THRESHOLD:
            st.success(f"Prediction: With Mask (Confidence: {mask_prob:.4f})")
            class_index = 1
        else:
            st.error(f"Prediction: Without Mask (Confidence: {no_mask_prob:.4f})")
            class_index = 0

        try:
            heatmap = make_gradcam_heatmap(processed, model, class_index)

            if heatmap is None:
                raise ValueError("Grad-CAM failed")

            cam_image = overlay_gradcam(image_bgr, heatmap)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

            metrics = gradcam_metrics(heatmap)

            st.subheader("Grad-CAM Visualization")
            st.image(cam_image, width="stretch")

            st.subheader("Numerical Explanation")
            st.table({
                "Metric": list(metrics.keys()),
                "Value": [
                    f"{v * 100:.2f} %" if "Contribution" in k or "Ratio" in k else f"{v:.4f}"
                    for k, v in metrics.items()
                ]
            })

        except Exception:
            st.warning(
                "Grad-CAM explanation could not be generated for this image. "
                "The prediction itself remains valid."
            )

# ==================================================
# WEBCAM SNAPSHOT
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

        mask_prob = prediction[0][1]
        no_mask_prob = prediction[0][0]

        if mask_prob >= THRESHOLD:
            st.success(f"Prediction: With Mask (Confidence: {mask_prob:.4f})")
            class_index = 1
        else:
            st.error(f"Prediction: Without Mask (Confidence: {no_mask_prob:.4f})")
            class_index = 0

        try:
            heatmap = make_gradcam_heatmap(processed, model, class_index)

            if heatmap is None:
                raise ValueError("Grad-CAM failed")

            cam_image = overlay_gradcam(image_bgr, heatmap)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

            metrics = gradcam_metrics(heatmap)

            st.subheader("Grad-CAM Visualization")
            st.image(cam_image, width="stretch")

            st.subheader("Numerical Explanation")
            st.table({
                "Metric": list(metrics.keys()),
                "Value": [
                    f"{v * 100:.2f} %" if "Contribution" in k or "Ratio" in k else f"{v:.4f}"
                    for k, v in metrics.items()
                ]
            })

        except Exception:
            st.warning(
                "Grad-CAM explanation could not be generated for this image. "
                "The prediction itself remains valid."
            )

# ==================================================
# LIVE VIDEO (LOCAL ONLY)
# ==================================================
elif mode == "Live Video":

    start = st.button("Start Camera")
    stop = st.button("Stop Camera")
    frame_window = st.empty()

    if start:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed = preprocess_image(frame)
            prediction = model.predict(processed)

            mask_prob = prediction[0][1]
            no_mask_prob = prediction[0][0]

            if mask_prob >= THRESHOLD:
                label = f"With Mask ({mask_prob:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Without Mask ({no_mask_prob:.2f})"
                color = (0, 0, 255)

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame, width="stretch")

        cap.release()
