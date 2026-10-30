Handwritten Digit Recognition using CNN

A Deep Learning project that recognizes handwritten digits (0–9) using the MNIST Dataset and a Convolutional Neural Network (CNN). The project includes model training, evaluation, and a web application where users can draw digits and get real-time predictions.

Features

- Handwritten digit recognition (0–9).
- CNN-based image classification.
- MNIST dataset for model training and testing.
- Interactive drawing canvas.
- Real-time digit prediction.
- Prediction confidence percentage.
- Probability scores for all 10 digits.
- Model evaluation dashboard.
- Accuracy, Precision, Recall and F1-Score.
- Confusion Matrix visualization.

Tech Stack

Technology| Purpose
Python| Programming Language
TensorFlow / Keras| Deep Learning
CNN| Image Classification
MNIST| Dataset
NumPy| Numerical Operations
Matplotlib| Data Visualization
Scikit-learn| Model Evaluation
Flask| Backend Development
HTML| Web Structure
CSS| Web Styling
JavaScript| Frontend Functionality
Pillow| Image Processing

Project Structure

handwritten-digit-recognition/
│
├── data/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_cnn_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/
│   ├── mnist_cnn.keras
│   └── evaluation_metrics.json
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── .gitignore
└── README.md

Dataset Information

This project uses the MNIST Handwritten Digit Dataset.

Dataset Details:

- Total Images: 70,000
- Training Images: 60,000
- Testing Images: 10,000
- Image Dimensions: 28 × 28 Pixels
- Image Format: Grayscale
- Number of Classes: 10 (0–9)

The dataset is loaded using TensorFlow's built-in MNIST dataset loader.

Data Preprocessing

The following preprocessing operations are performed:

1. Loading the MNIST dataset.
2. Exploring image dimensions and labels.
3. Visualizing handwritten digit samples.
4. Checking pixel intensity values.
5. Normalizing pixel values from 0–255 to 0–1.
6. Adding a channel dimension for CNN input.
7. Splitting training data into training and validation sets.

CNN Model Architecture

The project uses a Convolutional Neural Network with the following architecture:

1. Input Layer – 28 × 28 × 1
2. Conv2D – 32 Filters, ReLU Activation
3. MaxPooling2D
4. Conv2D – 64 Filters, ReLU Activation
5. MaxPooling2D
6. Flatten Layer
7. Dense Layer – 128 Neurons, ReLU Activation
8. Output Layer – 10 Neurons, Softmax Activation

Model Compilation

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Early Stopping: Used to monitor validation loss.

Model Evaluation

The trained model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro Average
- Weighted Average
- Classification Report
- Confusion Matrix

Evaluation results are stored in "models/evaluation_metrics.json".

Web Application

The project includes a Flask-based web application.

Users can:

1. Draw a handwritten digit using the drawing canvas.
2. Click the Predict Digit button.
3. Get the predicted digit.
4. View the prediction confidence.
5. View probability scores for all 10 digits.
6. Check the model's evaluation metrics.

Installation and Setup

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition.git

2. Navigate to the Project Directory

cd handwritten-digit-recognition

3. Install Required Libraries

python -m pip install -r requirements.txt

4. Run the Flask Backend

python backend/app.py

The backend will run at:

http://127.0.0.1:5000 // i will deploy it later please ignore it

5. Run the Frontend

Open "frontend/index.html" using the VS Code Live Server extension.

Make sure the Flask backend is running while using the web application.

Note: The trained model file and evaluation metrics JSON must be present inside the "models" folder. If they are not available, run the training and evaluation notebooks to generate them.

Future Improvements

- Improve prediction accuracy for different handwriting styles.
- Add prediction history.
- Improve the user interface.
- Add more model performance visualizations.
- Deploy the application online.

Author

Gauri Papnai

---

If you find this project useful, don't forget to ⭐ the repository!