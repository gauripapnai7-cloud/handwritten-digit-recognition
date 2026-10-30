// ========================================
// 1. GET HTML ELEMENTS
// ========================================

// Get the drawing canvas
const canvas = document.getElementById("drawingCanvas");

// Get the 2D drawing context
const ctx = canvas.getContext("2d");

// Get buttons
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

// Get prediction result elements
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const statusElement = document.getElementById("status");

// Get probability container
const probabilitiesElement = document.getElementById("probabilities");


// ========================================
// 2. CANVAS INITIALIZATION
// ========================================

// Set white background
ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Configure drawing color
ctx.strokeStyle = "#000000";

// Configure drawing thickness
ctx.lineWidth = 15;

// Make drawing lines smooth
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Track whether the user is drawing
let isDrawing = false;


// ========================================
// 3. GET MOUSE / TOUCH POSITION
// ========================================

function getPosition(event) {

    // Get canvas position and displayed dimensions
    const rect = canvas.getBoundingClientRect();

    // Calculate position relative to the canvas
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

    // Prevent default browser behavior
    event.preventDefault();

    // Mark drawing as active
    isDrawing = true;

    // Get current pointer position
    const position = getPosition(event);

    // Start a new drawing path
    ctx.beginPath();

    // Move pen to current position
    ctx.moveTo(position.x, position.y);

    // Draw a small dot for a single click/tap
    ctx.lineTo(position.x + 0.1, position.y + 0.1);
    ctx.stroke();
}


// ========================================
// 5. DRAW ON CANVAS
// ========================================

function draw(event) {

    // Stop if the user is not drawing
    if (!isDrawing) return;

    // Get current pointer position
    const position = getPosition(event);

    // Draw a line to the current position
    ctx.lineTo(position.x, position.y);

    // Render the line
    ctx.stroke();
}


// ========================================
// 6. STOP DRAWING
// ========================================

function stopDrawing() {

    // Mark drawing as inactive
    isDrawing = false;

    // End current drawing path
    ctx.closePath();
}


// ========================================
// 7. ATTACH DRAWING EVENTS
// ========================================

// Pointer events support mouse, touch and stylus
canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);

// Stop drawing when pointer is released
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);


// ========================================
// 8. CLEAR CANVAS
// ========================================

clearBtn.addEventListener("click", function() {

    // Clear the entire canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Restore white background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Reset prediction result
    predictionElement.textContent = "?";

    // Reset confidence
    confidenceElement.textContent = "Confidence: --";

    // Clear status
    statusElement.textContent = "";

    // Reset probabilities
    probabilitiesElement.textContent =
        "Draw a digit to see predictions.";

});


// ========================================
// 9. PREDICT DIGIT
// ========================================

predictBtn.addEventListener("click", async function() {

    // Convert canvas image into Base64 PNG data
    const imageData = canvas.toDataURL("image/png");

    // Update status
    statusElement.textContent = "Predicting...";

    // Disable button while prediction is running
    predictBtn.disabled = true;

    try {

        // Send drawing image to Flask backend
        const response = await fetch(
            "http://127.0.0.1:5000/api/predict",
            {
                method: "POST",

                // Specify JSON request format
                headers: {
                    "Content-Type": "application/json"
                },

                // Send image data
                body: JSON.stringify({
                    image: imageData
                })
            }
        );

        // Convert server response into JSON
        const result = await response.json();

        // Handle backend errors
        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        // Display predicted digit
        predictionElement.textContent = result.prediction;

        // Display prediction confidence
        confidenceElement.textContent =
            `Confidence: ${result.confidence}%`;

        // Update status
        statusElement.textContent = "Prediction successful!";

        // Display probability bars
        displayProbabilities(result.probabilities);

    } catch (error) {

        // Display error message
        statusElement.textContent = error.message;

        console.error("Prediction error:", error);

    } finally {

        // Enable predict button again
        predictBtn.disabled = false;
    }

});


// ========================================
// 10. DISPLAY PROBABILITIES
// ========================================

function displayProbabilities(probabilities) {

    // Clear previous probability results
    probabilitiesElement.innerHTML = "";

    // Loop through all 10 digit probabilities
    probabilities.forEach((probability, digit) => {

        // Create a row for each digit
        const row = document.createElement("div");

        row.className = "probability-row";

        // Add digit label and probability
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

        // Add row to the probability container
        probabilitiesElement.appendChild(row);

    });

}


// ========================================
// 11. LOAD MODEL EVALUATION METRICS
// ========================================

async function loadMetrics() {

    try {

        // Request evaluation metrics from Flask backend
        const response = await fetch(
            "http://127.0.0.1:5000/api/metrics"
        );

        // Convert response to JSON
        const metrics = await response.json();

        // Handle API errors
        if (!response.ok) {
            throw new Error(metrics.error || "Unable to load metrics.");
        }

        // Convert decimal values into percentages
        document.getElementById("accuracy").textContent =
            (metrics.accuracy * 100).toFixed(2) + "%";

        document.getElementById("precision").textContent =
            (metrics.macro_precision * 100).toFixed(2) + "%";

        document.getElementById("recall").textContent =
            (metrics.macro_recall * 100).toFixed(2) + "%";

        document.getElementById("f1score").textContent =
            (metrics.macro_f1_score * 100).toFixed(2) + "%";

    } catch (error) {

        // Display errors in browser console
        console.error("Metrics loading error:", error);

    }

}


// ========================================
// 12. LOAD METRICS WHEN PAGE OPENS
// ========================================

loadMetrics();