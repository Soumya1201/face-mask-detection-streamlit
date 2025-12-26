import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# -------------------------------------------------
# Detect cloud vs local deployment
# -------------------------------------------------
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "false") == "true"

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection with Explainability",
    layout="centered"
)

st.title("Face Mask Detection with Explainability")
st.write("Image upload, webcam snapshot, and live video face mask detection.")

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_mobilenetv2.keras")

model = load_model()

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_image(image_bgr):
    image = cv2.resize(image_bgr, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------------------------
# Grad-CAM computation (cloud-safe)
# -------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_channel = tf.gather(predictions, class_index, axis=1)

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    # Avoid division by zero
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return None

    heatmap /= max_val
    return heatmap.numpy()

# -------------------------------------------------
# Grad-CAM numerical metrics
# -------------------------------------------------
def gradcam_metrics(heatmap):

    total = np.sum(heatmap)
    if total == 0:
        raise ValueError("Empty heatmap")

    mean_activation = np.mean(heatmap)
    max_activation = np.max(heatmap)
    high_activation_ratio = np.sum(heatmap > 0.6) / heatmap.size

    h = heatmap.shape[0]
    upper_face = heatmap[:h // 2, :]
    lower_face = heatmap[h // 2:, :]

    upper_contribution = np.sum(upper_face) / total
    lower_contribution = np.sum(lower_face) / total

    red_ratio = np.sum(heatmap >= 0.66) / heatmap.size
    yellow_ratio = np.sum((heatmap >= 0.33) & (heatmap < 0.66)) / heatmap.size
    blue_ratio = np.sum(heatmap < 0.33) / heatmap.size

    return {
        "Mean Activation": mean_activation,
        "Max Activation": max_activation,
        "High Activation Ratio (>0.6)": high_activation_ratio,
        "Upper Face Contribution": upper_contribution,
        "Lower Face Contribution": lower_contribution,
        "Red Region Contribution (High Evidence)": red_ratio,
        "Yellow Region Contribution (Moderate Evidence)": yellow_ratio,
        "Blue Region Contribution (Low Evidence)": blue_ratio
    }

# -------------------------------------------------
# Overlay Grad-CAM
# -------------------------------------------------
def overlay_gradcam(image_bgr, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

# -------------------------------------------------
# App configuration
# -------------------------------------------------
THRESHOLD = 0.7

# IMPORTANT FIX: better MobileNetV2 layer
LAST_CONV_LAYER = "block_13_expand_relu"

if IS_CLOUD:
    mode = st.radio("Select Input Mode", ("Upload Image", "Webcam Snapshot"))
    st.info("Live video mode is available only in local deployment.")
else:
    mode = st.radio("Select Input Mode", ("Upload Image", "Webcam Snapshot", "Live Video"))

# =================================================
# IMAGE UPLOAD MODE
# =================================================
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

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

        # ---------- SAFE GRAD-CAM ----------
        try:
            heatmap = make_gradcam_heatmap(
                processed, model, LAST_CONV_LAYER, class_index
            )

            if heatmap is None:
                raise ValueError("Grad-CAM heatmap is empty")

            metrics = gradcam_metrics(heatmap)
            cam_image = overlay_gradcam(image_bgr, heatmap)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

            st.subheader("Grad-CAM Visualization")
            st.image(cam_image, use_container_width=True)

            st.subheader("Numerical Explanation (Grad-CAM Metrics)")
            st.table({
                "Metric": list(metrics.keys()),
                "Value": [
                    f"{v * 100:.2f} %" if "Contribution" in k or "Ratio" in k else f"{v:.4f}"
                    for k, v in metrics.items()
                ]
            })

        except Exception:
            st.warning(
                "Grad-CAM explanation is unavailable for this prediction due to "
                "gradient saturation. The prediction itself remains valid."
            )

# =================================================
# WEBCAM SNAPSHOT MODE
# =================================================
elif mode == "Webcam Snapshot":

    webcam_image = st.camera_input("Capture image")

    if webcam_image is not None:
        image = Image.open(webcam_image).convert("RGB")
        st.image(image, use_container_width=True)

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
            heatmap = make_gradcam_heatmap(
                processed, model, LAST_CONV_LAYER, class_index
            )

            if heatmap is None:
                raise ValueError("Grad-CAM heatmap is empty")

            metrics = gradcam_metrics(heatmap)
            cam_image = overlay_gradcam(image_bgr, heatmap)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

            st.subheader("Grad-CAM Visualization")
            st.image(cam_image, use_container_width=True)

            st.subheader("Numerical Explanation (Grad-CAM Metrics)")
            st.table({
                "Metric": list(metrics.keys()),
                "Value": [
                    f"{v * 100:.2f} %" if "Contribution" in k or "Ratio" in k else f"{v:.4f}"
                    for k, v in metrics.items()
                ]
            })

        except Exception:
            st.warning(
                "Grad-CAM explanation is unavailable for this prediction due to "
                "gradient saturation. The prediction itself remains valid."
            )

# =================================================
# LIVE VIDEO MODE (LOCAL ONLY)
# =================================================
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

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame, use_container_width=True)

        cap.release()
