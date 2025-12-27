## Overview
This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
The system takes an image of a handwritten digit (0–9) and predicts the digit using a trained deep learning model.

## Dataset Used
-MNIST Dataset
-70,000 grayscale images of handwritten digits
-Image size: 28 × 28
-Classes: 0–9
-Loaded automatically using TensorFlow/Keras

``` text
## Project Structure
│
├── train_model.py          # CNN training script
├── digit_cnn_model.h5      # Saved trained model
├── test_model.py           # Model verification script
├── app_gradio.py           # Gradio web interface
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
## Technologies Used
-Python
-TensorFlow / Keras
-NumPy
-Pillow
-Gradio

## How the Project Works
-The CNN is trained on MNIST images (train_model.py)
-The trained model is saved as digit_cnn_model.h5
-The model is tested using unseen MNIST test images (test_model.py)
-A Gradio UI allows users to upload or draw digits (app_gradio.py)
-The model predicts the digit in real time

## How to Run the Project
-Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```
-Train the Model
```bash
python3 train_model.py
```
-Test the Model
```bash
python3 test_model.py
```
-Run the Web App
```bash
python3 app_gradio.py
```

