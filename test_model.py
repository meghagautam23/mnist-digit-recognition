import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

model = load_model("digit_cnn_model.h5")

(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

pred = model.predict(X_test[:1])
print("Predicted:", np.argmax(pred))
print("Actual:", y_test[0])
