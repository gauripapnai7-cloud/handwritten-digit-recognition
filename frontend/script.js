
// ========================================
// 1. GET HTML ELEMENTS
// ========================================

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const statusElement = document.getElementById("status");

const probabilitiesElement = document.getElementById("probabilities");
const predictionMode = document.getElementById("predictionMode");


// ========================================
// 2. CANVAS INITIALIZATION
// ========================================

function resetCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

resetCanvas();

ctx.strokeStyle = "#000000";
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let isDrawing = false;
let hasDrawn = false;


// ========================================
// 3. GET POINTER POSITION
// ========================================

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();

    const x = (event.clientX - rect.left) *
        (canvas.width / rect.width);

    const y = (event.clientY - rect.top) *
        (canvas.height / rect.height);

    return { x, y };
}


// ========================================
// 4. START DRAWING
// ========================================

function startDrawing(event) {
    event.preventDefault();

    isDrawing = true;
    hasDrawn = true;

    const position = getPosition(event);

    ctx.beginPath();
    ctx.moveTo(position.x, position.y);

    ctx.lineTo(position.x + 0.1, position.y + 0.1);
    ctx.stroke();
}


// ========================================
// 5. DRAW
// ========================================

function draw(event) {
    if (!isDrawing) return;

    const position = getPosition(event);

    ctx.lineTo(position.x, position.y);
    ctx.stroke();
}


// ========================================
// 6. STOP DRAWING
// ========================================

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
}


// ========================================
// 7. DRAWING EVENTS
// ========================================

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);


// ========================================
// 8. CLEAR CANVAS
// ========================================

clearBtn.addEventListener("click", function() {

    resetCanvas();

    hasDrawn = false;

    predictionElement.textContent = "?";
    predictionElement.style.whiteSpace = "normal";

    confidenceElement.textContent = "Confidence: --";
    statusElement.textContent = "";

    probabilitiesElement.textContent =
        "Draw a digit to see predictions.";
});


// ========================================
// 9. PREDICTION MODE
// ========================================

predictionMode.addEventListener("change", function() {

    predictionElement.textContent = "?";
    predictionElement.style.whiteSpace = "normal";

    confidenceElement.textContent = "Confidence: --";
    statusElement.textContent = "";

    probabilitiesElement.textContent =
        predictionMode.value === "multiple"
        ? "Draw multiple digits in one or more rows."
        : "Draw a digit to see predictions.";

});


// ========================================
// 10. PREDICT DIGITS
// ========================================

predictBtn.addEventListener("click", async function() {

    if (!hasDrawn) {
        statusElement.textContent = "Please draw something first.";
        return;
    }

    const mode = predictionMode.value;

    const endpoint = mode === "multiple"
        ? "/api/predict-multiple"
        : "/api/predict";

    const imageData = canvas.toDataURL("image/png");

    statusElement.textContent = "Predicting...";

    predictBtn.disabled = true;

    try {

        const response = await fetch(endpoint, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: imageData
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        // SINGLE DIGIT RESULT
        if (mode === "single") {

            predictionElement.textContent = result.prediction;
            predictionElement.style.whiteSpace = "normal";

            confidenceElement.textContent =
                `Confidence: ${result.confidence}%`;

            displayProbabilities(result.probabilities);

        }

        // MULTIPLE DIGIT RESULT
        else {

            // Preserve multiple row formatting
            predictionElement.textContent = result.prediction;
            predictionElement.style.whiteSpace = "pre-line";

            const averageConfidence = result.digits.length
                ? result.digits.reduce(
                    (sum, item) => sum + item.confidence, 0
                  ) / result.digits.length
                : 0;

            confidenceElement.textContent =
                `Average Confidence: ${averageConfidence.toFixed(2)}%`;

            displayMultipleDigits(result);

        }

        statusElement.textContent = "Prediction successful!";

    } catch (error) {

        predictionElement.textContent = "?";
        confidenceElement.textContent = "Confidence: --";

        statusElement.textContent = error.message;

        console.error("Prediction error:", error);

    } finally {

        predictBtn.disabled = false;

    }

});


// ========================================
// 11. DISPLAY SINGLE DIGIT PROBABILITIES
// ========================================

function displayProbabilities(probabilities) {

    probabilitiesElement.innerHTML = "";

    probabilities.forEach((probability, digit) => {

        const row = document.createElement("div");

        row.className = "probability-row";

        row.innerHTML = `
            <div class="probability-info">
                <span>Digit ${digit}</span>
                <span>${probability.toFixed(2)}%</span>
            </div>

            <div class="probability-track">
                <div class="probability-fill"
                     style="width: ${Math.min(100, Math.max(0, probability))}%">
                </div>
            </div>
        `;

        probabilitiesElement.appendChild(row);

    });

}


// ========================================
// 12. DISPLAY MULTIPLE DIGIT RESULTS
// ========================================

function displayMultipleDigits(result) {

    probabilitiesElement.innerHTML = "";

    const heading = document.createElement("h3");
    heading.textContent = "Individual Digit Predictions";

    probabilitiesElement.appendChild(heading);

    // Get rows and individual digit predictions
    const rows = result.rows || [result.prediction];
    const digits = result.digits || [];

    let digitIndex = 0;

    rows.forEach((rowDigits, rowIndex) => {

        const rowHeading = document.createElement("h4");

        rowHeading.textContent = `Row ${rowIndex + 1}: ${rowDigits}`;

        probabilitiesElement.appendChild(rowHeading);

        // Show confidence for every digit in this row
        for (let i = 0; i < rowDigits.length; i++) {

            const item = digits[digitIndex];

            if (!item) break;

            const row = document.createElement("div");

            row.className = "probability-row";

            row.innerHTML = `
                <div class="probability-info">
                    <span>Digit ${digitIndex + 1}: ${item.digit}</span>
                    <span>${item.confidence.toFixed(2)}%</span>
                </div>

                <div class="probability-track">
                    <div class="probability-fill"
                         style="width: ${Math.min(100, Math.max(0, item.confidence))}%">
                    </div>
                </div>
            `;

            probabilitiesElement.appendChild(row);

            digitIndex++;
        }

    });

}


// ========================================
// 13. LOAD MODEL EVALUATION METRICS
// ========================================

async function loadMetrics() {

    try {

        const response = await fetch("/api/metrics");

        const metrics = await response.json();

        if (!response.ok) {
            throw new Error(metrics.error || "Unable to load metrics.");
        }

        document.getElementById("accuracy").textContent =
            (metrics.accuracy * 100).toFixed(2) + "%";

        document.getElementById("precision").textContent =
            (metrics.macro_precision * 100).toFixed(2) + "%";

        document.getElementById("recall").textContent =
            (metrics.macro_recall * 100).toFixed(2) + "%";

        document.getElementById("f1score").textContent =
            (metrics.macro_f1_score * 100).toFixed(2) + "%";

    } catch (error) {

        console.error("Metrics loading error:", error);

    }

}


// ========================================
// 14. LOAD METRICS
// ========================================

loadMetrics();
