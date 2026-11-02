
import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from predict import predict_multiple_digits


# -------------------------------
# PROJECT PATHS
# -------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_cnn.keras")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="DigitAI | Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)


# -------------------------------
# CSS DESIGN
# -------------------------------

st.markdown("""
<style>

* {
    box-sizing: border-box;
}

.stApp {
    background: #0b1020;
    color: #ffffff;
}

.block-container {
    max-width: 1200px;
    padding: 30px !important;
    padding-top: 30px !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.logo {
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #ffffff;
}

.logo span {
    color: #38bdf8;
}

.tagline {
    text-align: center;
    color: #94a3b8;
    margin-top: 8px;
    margin-bottom: 40px;
    font-size: 15px;
}

h2, h3 {
    color: #ffffff !important;
}

p, label {
    color: #cbd5e1;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 25px;
}

div[data-testid="stSelectbox"] label {
    color: #ffffff !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #111827;
    border-color: #334155;
    color: white;
}

div[data-testid="stSelectbox"] span {
    color: white;
}

/* CANVAS ALIGNMENT */

div[data-testid="stCanvas"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

div[data-testid="stCanvas"] iframe {
    margin-left: auto !important;
    margin-right: auto !important;
}

/* BUTTONS */

.stButton > button {
    border: none;
    border-radius: 10px;
    padding: 13px 24px;
    min-height: 43px;
    font-size: 15px;
    font-weight: 600;
    transition: 0.25s ease;
}

.stButton > button[kind="secondary"] {
    background: #334155;
    color: white;
}

.stButton > button[kind="secondary"]:hover {
    background: #475569;
    color: white;
}

.stButton > button[kind="primary"] {
    background: #0284c7;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: #0369a1;
    color: white;
    transform: translateY(-2px);
}

/* RESULT BOX */

.result-card {
    width: 100%;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: #111827;
    border-radius: 15px;
    padding: 15px 20px;
    margin-top: 25px;
    margin-bottom: 15px;
    position: relative;
    overflow: hidden;


     
    
    

     
     
}

.result-heading {
    color: #94a3b8;
    font-size: 20px;
    line-height: 1.5;
    margin-bottom: 20pxx;
    
}

 .result-number {
    width: 100%;
    font-size: 90px;
    font-weight: 800;
    color: #38bdf8;
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow-wrap: anywhere;.\venv\Scripts\Activate
    white-space: pre-line;
    line-height: 1.2;
    margin: 10px auto; 
    transform: translateY(-8px);
}

#prediction {
    transform: translateY(-20px);
    font-size: 90px;
}

#prediction.predicted {
    transform: translateY(-20px);
    font-size: 90px;
}

.confidence {
    color: #4ade80;
    text-align: center;
    font-weight: 600;
    margin-top: 12px;
}

.status {
    color: #94a3b8;
    text-align: center;
    font-size: 13px;
    margin-top: 12px;
}

/* PROBABILITIES */

.probability-title {
    color: white;
    font-size: 19px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.probability-info {
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 5px;
}

.probability-track {
    height: 8px;
    background: #334155;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 12px;
}

.probability-fill {
    height: 100%;
    background: #38bdf8;
    border-radius: 20px;
}

/* METRICS */

.metric-card {
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin-bottom: 15px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #38bdf8;
    font-size: 28px;
    font-weight: 700;
    margin-top: 12px;
}

/* MODEL INFORMATION */

.model-info {
    background: #111827;
    padding: 22px;
    border-radius: 14px;
    margin-top: 25px;
}

.model-info h3 {
    margin-bottom: 15px;
}

.model-info p {
    color: #94a3b8;
    font-size: 14px;
    margin: 12px 0;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 35px;
    padding: 20px;
}

@media (max-width: 850px) {

    .block-container {
        padding: 15px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 22px;
    }

}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------

st.markdown("""
<div class="logo">Digit<span>AI</span></div>
<div class="tagline">Handwritten Digit Recognition</div>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# -------------------------------
# LOAD METRICS
# -------------------------------

metrics = {}

if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)
    except Exception:
        metrics = {}


# -------------------------------
# PREPROCESS IMAGE
# -------------------------------

def preprocess_image(image):

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)

    bbox = image.getbbox()

    if bbox:
        image = image.crop(bbox)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    image_array = np.array(canvas) / 255.0

    return image_array.reshape(1, 28, 28, 1)


# -------------------------------
# SESSION STATE
# -------------------------------

defaults = {
    "canvas_key": 0,
    "prediction_result": "?",
    "confidence_result": None,
    "probability_result": None,
    "multiple_details": None,
    "status_result": "",
    "last_mode": "Single Digit"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------
# MAIN LAYOUT
# -------------------------------

left_col, right_col = st.columns([1, 1], gap="large")


# -------------------------------
# LEFT SECTION
# -------------------------------

with left_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Draw a Digit</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">Draw any digit or number</div>',
            unsafe_allow_html=True
        )

        mode = st.selectbox(
            "Prediction Mode:",
            ["Single Digit", "Multiple Digits"],
            key="prediction_mode"
        )

        if mode != st.session_state.last_mode:

            st.session_state.prediction_result = "?"
            st.session_state.confidence_result = None
            st.session_state.probability_result = None
            st.session_state.multiple_details = None
            st.session_state.status_result = ""
            st.session_state.last_mode = mode

        # CENTERED DRAWING CANVAS

        canvas_left, canvas_center, canvas_right = st.columns(
            [1, 2, 1],
            gap="small"
        )

        with canvas_center:

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#ffffff",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"drawing_canvas_{st.session_state.canvas_key}",
                return_image_data=True
            )

        # BUTTONS

        clear_col, predict_col = st.columns(2)

        with clear_col:

            clear_clicked = st.button(
                "Clear",
                use_container_width=True,
                type="secondary"
            )

        with predict_col:

            predict_clicked = st.button(
                "Predict Digit",
                use_container_width=True,
                type="primary"
            )

        # CLEAR BUTTON

        if clear_clicked:

            st.session_state.canvas_key += 1
            st.session_state.prediction_result = "?"
            st.session_state.confidence_result = None
            st.session_state.probability_result = None
            st.session_state.multiple_details = None
            st.session_state.status_result = ""

            st.rerun()

        # PREDICTION

        if predict_clicked:

            if canvas_result.image_data is None:

                st.session_state.status_result = (
                    "Please draw something first."
                )

            elif not np.any(
                canvas_result.image_data[:, :, :3] < 245
            ):

                st.session_state.status_result = (
                    "Please draw a digit first."
                )

            else:

                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("RGB")

                try:

                    with st.spinner("Predicting..."):

                        if mode == "Single Digit":

                            processed_image = preprocess_image(image)

                            predictions = model.predict(
                                processed_image,
                                verbose=0
                            )

                            probabilities = predictions[0] * 100

                            digit = int(np.argmax(probabilities))

                            confidence = float(
                                np.max(probabilities)
                            )

                            st.session_state.prediction_result = str(digit)

                            st.session_state.confidence_result = confidence

                            st.session_state.probability_result = (
                                probabilities.tolist()
                            )

                            st.session_state.multiple_details = None

                        else:

                            image_array = np.array(
                                image.convert("L")
                            )

                            result = predict_multiple_digits(
                                image_array,
                                model
                            )

                            if isinstance(result, dict):

                                st.session_state.prediction_result = str(
                                    result.get("prediction", "")
                                )

                                digits = result.get("digits", [])

                                if digits:

                                    average_confidence = sum(
                                        float(item["confidence"])
                                        for item in digits
                                    ) / len(digits)

                                else:
                                    average_confidence = 0

                                st.session_state.confidence_result = (
                                    average_confidence
                                )

                                st.session_state.multiple_details = result
                                st.session_state.probability_result = None

                            else:

                                st.session_state.prediction_result = str(result)
                                st.session_state.confidence_result = 0
                                st.session_state.multiple_details = None
                                st.session_state.probability_result = None

                    st.session_state.status_result = (
                        "Prediction successful!"
                    )

                except Exception as e:

                    st.session_state.prediction_result = "?"
                    st.session_state.confidence_result = None
                    st.session_state.probability_result = None
                    st.session_state.multiple_details = None

                    st.session_state.status_result = (
                        f"Prediction failed: {e}"
                    )

        # RESULT BOX

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-heading">
                    Predicted Digit / Number
                </div>
                <div class="result-number">
                    {st.session_state.prediction_result}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # CONFIDENCE

        if st.session_state.confidence_result is not None:

            confidence_label = (
                "Average Confidence"
                if mode == "Multiple Digits"
                else "Confidence"
            )

            st.markdown(
                f"""
                <div class="confidence">
                    {confidence_label}:
                    {st.session_state.confidence_result:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # STATUS

        if st.session_state.status_result:

            st.markdown(
                f'<div class="status">{st.session_state.status_result}</div>',
                unsafe_allow_html=True
            )

        # SINGLE DIGIT PROBABILITIES

        if st.session_state.probability_result is not None:

            st.markdown(
                '<div class="probability-title">Digit Probabilities</div>',
                unsafe_allow_html=True
            )

            for digit, probability in enumerate(
                st.session_state.probability_result
            ):

                probability = float(probability)

                st.markdown(
                    f"""
                    <div class="probability-info">
                        <span>Digit {digit}</span>
                        <span>{probability:.2f}%</span>
                    </div>

                    <div class="probability-track">
                        <div class="probability-fill"
                             style="width: {min(100, max(0, probability))}%">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # MULTIPLE DIGIT PREDICTIONS

        if st.session_state.multiple_details is not None:

            details = st.session_state.multiple_details

            st.markdown(
                '<div class="probability-title">Individual Digit Predictions</div>',
                unsafe_allow_html=True
            )

            rows = details.get(
                "rows",
                [details.get("prediction", "")]
            )

            digits = details.get("digits", [])

            digit_index = 0

            for row_index, row_digits in enumerate(rows):

                st.markdown(
                    f"**Row {row_index + 1}: {row_digits}**"
                )

                for _ in str(row_digits):

                    if digit_index >= len(digits):
                        break

                    item = digits[digit_index]

                    digit_value = item.get("digit", "")

                    confidence_value = float(
                        item.get("confidence", 0)
                    )

                    st.markdown(
                        f"""
                        <div class="probability-info">
                            <span>Digit {digit_index + 1}: {digit_value}</span>
                            <span>{confidence_value:.2f}%</span>
                        </div>

                        <div class="probability-track">
                            <div class="probability-fill"
                                 style="width: {min(100, max(0, confidence_value))}%">
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    digit_index += 1


# -------------------------------
# RIGHT SECTION
# -------------------------------

with right_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Model Performance</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">CNN model evaluation metrics</div>',
            unsafe_allow_html=True
        )

        accuracy = metrics.get("accuracy")
        precision = metrics.get("macro_precision")
        recall = metrics.get("macro_recall")
        f1 = metrics.get("macro_f1_score")

        metric_values = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1)
        ]

        for i in range(0, 4, 2):

            col1, col2 = st.columns(2)

            for col, item in zip(
                [col1, col2],
                metric_values[i:i + 2]
            ):

                name, value = item

                if isinstance(value, (int, float)):
                    display_value = f"{value * 100:.2f}%"
                else:
                    display_value = "--"

                with col:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value">{display_value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("""
        <div class="model-info">
            <h3>Model Information</h3>
            <p>Dataset: MNIST</p>
            <p>Architecture: Convolutional Neural Network</p>
            <p>Input Size: 28 × 28 pixels</p>
            <p>Output Classes: 10 Digits</p>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------
# FOOTER
# -------------------------------

st.markdown("""
<div class="footer">
    DigitAI | Built using TensorFlow, CNN and Flask
</div>
""", unsafe_allow_html=True)
