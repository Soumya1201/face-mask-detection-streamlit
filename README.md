# Face Mask Detection System

This project implements a deep learning–based Face Mask Detection System that classifies whether a person is wearing a face mask or not. The system is built using a Convolutional Neural Network (CNN) with transfer learning and is deployed as a web application using Streamlit.

---

## Deployed Application
 
The application is publicly accessible at the following link:

https://face-mask-detection-app-xyz.streamlit.app/

---

## Dataset

The model is trained using a publicly available dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data

### Dataset Description

- Two classes:
  - with_mask
  - without_mask
- Total images: approximately 7,500
- Nearly balanced distribution between the two classes
- Images include variations in lighting, pose, and background

---

## Model Overview

- Base architecture: MobileNetV2 (transfer learning)
- Input image size: 128 × 128 × 3
- Model architecture:
  - Pre-trained MobileNetV2 feature extractor
  - Global Average Pooling layer
  - Fully connected dense layer
  - Dropout layer for regularization
  - Softmax output layer with two classes
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

The model outputs probabilistic confidence scores for both classes.

---

## Features

- Image upload–based face mask detection
- Webcam snapshot–based prediction
- Displays prediction label along with confidence score
- Clean and responsive web interface
- Cloud-deployed and accessible via browser

---

## Technology Stack

- Programming Language: Python
- Deep Learning Framework: TensorFlow / Keras
- Web Framework: Streamlit
- Computer Vision: OpenCV
- Image Processing: Pillow (PIL)
- Numerical Computation: NumPy

---

## Project Structure

face-mask-detection-streamlit/
│
├── app.py Streamlit application
├── face_mask_mobilenetv2.keras Trained deep learning model
├── README.md Project documentation
├── requirements.txt Python dependencies




---

## How to Run the Project Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/Soumya1201/face-mask-detection-streamlit.git
cd face-mask-detection-streamlit
```

Step 2: Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
```

On Linux or macOS:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```
Step 3: Install required dependencies
```bash
pip install -r requirements.txt
```
Step 4: Run the Streamlit application
```bash
streamlit run app.py
```

The application will open automatically in the default web browser.

## Deployment

- The application is deployed using Streamlit Cloud. Any updates pushed to the GitHub repository automatically trigger redeployment.

- Advanced features such as live video streaming and explainability modules were excluded from deployment to ensure stability and compatibility with cloud environments.

## Notes

- Webcam snapshot functionality depends on browser permissions

- Model predictions are probabilistic and include confidence scores

- The system is intended for academic, learning, and demonstration purposes

## Author

Soumyadip Adhikaryjjj
