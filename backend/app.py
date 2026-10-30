# Import CORS to allow frontend requests
from flask_cors import CORS

# Import Flask modules for creating the backend server
from flask import Flask, request, jsonify

# Import TensorFlow for loading our trained CNN model
import tensorflow as tf

# Import NumPy for image array operations
import numpy as np

# Import Pillow for image processing
from PIL import Image, ImageOps

# Import OS for handling file paths
import os

# Import JSON for reading evaluation metrics
import json

# Import BytesIO for processing image data in memory
from io import BytesIO

# Create the Flask application
app = Flask(__name__)
CORS(app);

# Get the root project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the trained model path
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mnist_cnn.keras"
)

# Define the evaluation metrics path
METRICS_PATH = os.path.join(
    BASE_DIR, "models", "evaluation_metrics.json"
)

# Load the trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# Function to preprocess the user's drawing
def preprocess_image(image):

    # Convert image to grayscale
    image = image.convert("L")

    # Invert colors: white background becomes black
    # Black drawing becomes white, matching MNIST style
    image = ImageOps.invert(image)

    # Find the bounding box containing the drawn digit
    bbox = image.getbbox()

    # Return an error if the canvas is empty
    if bbox is None:
        raise ValueError("Please draw a digit first.")

    # Crop the image around the digit
    image = image.crop(bbox)

    # Resize the digit while maintaining its aspect ratio
    image.thumbnail((20, 20))

    # Create a blank 28x28 black image
    canvas = Image.new("L", (28, 28), 0)

    # Calculate position to center the digit
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    # Paste the resized digit into the center
    canvas.paste(image, (x_offset, y_offset))

    # Convert image into a NumPy array
    image_array = np.array(canvas, dtype="float32")

    # Normalize pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# API route for predicting handwritten digits
@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # Get image data sent by the frontend
        data = request.get_json()

        # Check whether image data exists
        if not data or "image" not in data:
            return jsonify({"error": "Image data is missing."}), 400

        # Remove the Data URL prefix
        image_data = data["image"].split(",", 1)[1]

        # Decode the image data
        import base64
        image_bytes = base64.b64decode(image_data)

        # Open the image using Pillow
        image = Image.open(BytesIO(image_bytes))

        # Preprocess the drawing for the CNN model
        processed_image = preprocess_image(image)

        # Generate prediction probabilities
        predictions = model.predict(processed_image, verbose=0)[0]

        # Find the digit with the highest probability
        predicted_digit = int(np.argmax(predictions))

        # Get confidence percentage
        confidence = float(np.max(predictions) * 100)

        # Return prediction results
        return jsonify({
            "prediction": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": [
                round(float(value) * 100, 2)
                for value in predictions
            ]
        })

    except ValueError as error:

        # Return validation errors
        return jsonify({"error": str(error)}), 400

    except Exception as error:

        # Return unexpected errors
        return jsonify({"error": str(error)}), 500


# API route for fetching evaluation metrics
@app.route("/api/metrics", methods=["GET"])
def get_metrics():

    # Check whether metrics file exists
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Evaluation metrics not found."}), 404

    # Read saved evaluation metrics
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    # Return metrics as JSON
    return jsonify(metrics)


# Start the Flask development server
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )