"Convert a .keras trained model to ONNX format"

import os
import yaml
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model


# CONFIG
# local config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
APP_PATH = cfg["app_data"]["local_path"]
MODEL = os.path.join(APP_PATH, cfg["app_data"]["model"])
ONNX = os.path.join(APP_PATH, cfg["app_data"]["onnx"])
BREEDS = cfg["app_data"]["breeds"]
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# *****************************************************
# IF NOT DONE, first load .keras model for future ONNX conversion
logging.info("\n👉 Model path: " + MODEL)
model = load_model(MODEL)
logging.info(model.summary())
logging.info(model.name)

# 🚧 remove useless data augmentation layer

# ONNX conversion with tf2onnx
import tf2onnx
import onnx

# 1. if needed, force / simulate the 'output_names' variable
model.output_names = ['output']
# 2. Specify an input signature (adjust for each model)
input_signature = [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)]
# 3. Convert Keras model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
# 4. Save ONNX model
onnx.save_model(onnx_model, ONNX)

# *****************************************************
# AFTER CONVERSION, test ONNX model with onnxruntime to check behaviour
import onnxruntime as ort
from tensorflow.keras.preprocessing import image
import numpy as np


# inputs
img_path = "app_data/chow/n02112137_9159.jpg"
img_size = (224, 224)
session = ort.InferenceSession(ONNX)

# preprocess image
# resize
img = image.load_img(img_path, target_size=img_size)
# array conversion
img_array = image.img_to_array(img)
# dimension add for batch
img_array = np.expand_dims(img_array, axis=0)

# model input names: useful?
input_name = session.get_inputs()[0].name
# inference
predictions = session.run(None, {input_name: img_array})[0][0]

# results
logging.info(f"👉 raw predictions: {predictions}")
# predict class #
predicted_class = np.argmax(predictions)
logging.info(f"👉 predicted class: {predicted_class}")
# predict dog breed
breed = BREEDS[predicted_class]
logging.info(f"👉 predicted breed: {breed}")
# confidence
confidence = predictions[predicted_class]
logging.info(f"👉 confidence: {confidence}")
# text output
print(f"I'm {confidence :.0%} sure this is a {breed}.")
