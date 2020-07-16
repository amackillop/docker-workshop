from flask import Flask, request
import requests
import tensorflow as tf
import numpy as np
import imghdr
from io import BytesIO


app = Flask(__name__)

MODEL = tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


@app.route("/predict", methods=["POST"])
def predict():
    url = request.json["url"]
    return handle_request(url)


def handle_request(url: str):
    img_bytes = download_image(url)
    img = parse_image(img_bytes)
    img_array = preprocess(img)
    predictions = predict(img_array)
    response = "\n".join(str(pred) for pred in predictions)
    return response


def download_image(url: str) -> bytes:
    """Download and verify image from given URL."""
    res = requests.get(url)
    res.raise_for_status()
    content = res.content

    # Weak check that the page content is actually an image.
    if imghdr.what(BytesIO(content)) is None:
        msg = f"Not a valid image at {url}."
        raise IOError(msg)

    return content


def parse_image(img_bytes: bytes) -> np.array:
    img = tf.image.decode_image(img_bytes, channels=3, dtype=tf.uint8)
    img = tf.image.resize_with_pad(img, target_width=224, target_height=224)
    return img


def preprocess(img: np.array) -> np.array:
    img = np.array([img])
    return tf.keras.applications.mobilenet.preprocess_input(img, data_format=None)


def predict(img_array: np.array):
    prediction = MODEL.predict(img_array)
    return tf.keras.applications.mobilenet.decode_predictions(prediction, top=5)[0]
