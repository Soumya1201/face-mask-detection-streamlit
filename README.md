Face Mask Detection System

A deep learning–based Face Mask Detection System that classifies whether a person is wearing a mask or not using a Convolutional Neural Network (CNN). The application is deployed using Streamlit and supports image upload and webcam snapshot inputs.

Deployed Application

🔗 Live App:
https://face-mask-detection-app-xyz.streamlit.app/

Dataset

The model is trained on a publicly available Kaggle dataset:

🔗 Dataset Link:
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data

Dataset Details

Two classes:

  with_mask
  
  without_mask

Total images: ~7,500+

Balanced dataset (nearly equal samples per class)

Model Overview

Base Model: MobileNetV2 (transfer learning)

Input Size: 128 × 128 × 3

Classifier Head:

Global Average Pooling

Fully Connected (Dense) layers

Softmax output (2 classes)

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Features

Upload an image and get mask detection result

Capture image using webcam (snapshot)

Displays confidence scores for both classes

Clean and responsive UI using Streamlit

Cloud-deployed and accessible via browser

Tech Stack

Programming Language: Python

Deep Learning: TensorFlow / Keras

Computer Vision: OpenCV

Web Framework: Streamlit

Image Processing: PIL, NumPy

Project Structure
face-mask-detection-streamlit/
│
├── app.py                         # Streamlit application
├── face_mask_mobilenetv2.keras    # Trained model
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies

How to Run Locally
Clone the repository

git clone https://github.com/Soumya1201/face-mask-detection-streamlit.git
cd face-mask-detection-streamlit

Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app.py

 Deployment

Deployed using Streamlit Cloud

Automatically redeploys on GitHub commits

Live video and Grad-CAM were excluded to ensure cloud stability

Notes

Webcam snapshot works on supported browsers

Model predictions are probabilistic; confidence scores are shown

Designed for academic, demo, and learning purposes

Author

Soumyadip Adhikary
Project – Face Mask Detection using Deep Learning
