import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("digit_cnn_model.h5")

def predict_digit(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    return int(np.argmax(prediction))

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Handwritten Digit Recognition",
    description="Upload a handwritten digit (0–9)"
)

interface.launch()
