import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import transform

# Load the model
model = tf.keras.models.load_model('model.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
