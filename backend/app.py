
# Import CORS
from flask_cors import CORS

# Import Flask modules
from flask import Flask, request, jsonify, send_from_directory

# Import TensorFlow
import tensorflow as tf

# Import NumPy
import numpy as np

# Import Pillow
from PIL import Image, ImageOps

# Import OS
import os

# Import JSON
import json

# Import BytesIO
from io import BytesIO

# Import multiple digit prediction function
from predict import predict_multiple_digits


# ==========================================
# PROJECT PATHS
# ==========================================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Frontend folder path
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Model file path
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mnist_cnn.keras"
)

# Evaluation metrics path
METRICS_PATH = os.path.join(
    BASE_DIR, "models", "evaluation_metrics.json"
)


# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

# Connect frontend folder with Flask
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

# Enable CORS
CORS(app)


# ==========================================
# LOAD CNN MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# ==========================================
# HOMEPAGE ROUTE
# ==========================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==========================================
# SINGLE DIGIT IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    # Convert image to grayscale
    image = image.convert("L")

    # Invert colors
    image = ImageOps.invert(image)

    # Find digit boundaries
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("Please draw a digit first.")

    # Crop digit
    image = image.crop(bbox)

    # Resize while maintaining aspect ratio
    image.thumbnail((20, 20))

    # Create 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    # Center digit
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Convert to NumPy array
    image_array = np.array(canvas, dtype="float32")

    # Normalize
    image_array = image_array / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# ==========================================
# SINGLE DIGIT PREDICTION API
# ==========================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "Image data is missing."
            }), 400

        import base64

        image_data = data["image"].split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(BytesIO(image_bytes))

        processed_image = preprocess_image(image)

        predictions = model.predict(
            processed_image, verbose=0
        )[0]

        predicted_digit = int(np.argmax(predictions))

        confidence = float(np.max(predictions) * 100)

        return jsonify({
            "prediction": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": [
                round(float(value) * 100, 2)
                for value in predictions
            ]
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ==========================================
# MULTIPLE DIGIT PREDICTION API
# ==========================================

@app.route("/api/predict-multiple", methods=["POST"])
def predict_multiple():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "Image data is missing."
            }), 400

        import base64

        image_data = data["image"].split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        # Open image in grayscale
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("L")

        image_array = np.array(image)

        # Predict multiple digits
        result = predict_multiple_digits(
            image_array, model
        )

        return jsonify(result)

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ==========================================
# EVALUATION METRICS API
# ==========================================

@app.route("/api/metrics", methods=["GET"])
def get_metrics():

    if not os.path.exists(METRICS_PATH):
        return jsonify({
            "error": "Evaluation metrics not found."
        }), 404

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return jsonify(metrics)


# ==========================================
# START FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
