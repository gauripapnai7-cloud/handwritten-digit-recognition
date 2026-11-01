
import numpy as np
import cv2


def predict_multiple_digits(image, model):
    """
    Horizontal aur multiple-row digits ko recognize karta hai.
    Example:
        32
        21
        92

    Output:
        32
        21
        92
    """

    # Image ko grayscale mein convert karo
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Black background aur white digits
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise remove karo
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel
    )

    # Connected digit regions find karo
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Chhote noise ko ignore karo
        if w < 5 or h < 10:
            continue

        digit_boxes.append((x, y, w, h))

    if not digit_boxes:
        raise ValueError("Image mein koi digit nahi mila.")

    # Top-to-bottom order mein rows identify karo
    digit_boxes.sort(key=lambda box: box[1])

    rows = []

    for box in digit_boxes:
        x, y, w, h = box
        center_y = y + h / 2

        matched_row = None

        # Existing row ke center ke aas-paas digit hai?
        for row in rows:
            row_center = np.mean([
                item[1] + item[3] / 2
                for item in row
            ])

            average_height = np.mean([
                item[3] for item in row
            ])

            if abs(center_y - row_center) < average_height * 0.5:
                matched_row = row
                break

        if matched_row is None:
            rows.append([box])
        else:
            matched_row.append(box)

    # Har row ko left-to-right arrange karo
    for row in rows:
        row.sort(key=lambda box: box[0])

    # Rows ko top-to-bottom arrange karo
    rows.sort(
        key=lambda row: min(box[1] for box in row)
    )

    all_results = []
    final_rows = []

    # Har row ke digits predict karo
    for row in rows:

        row_prediction = ""

        for x, y, w, h in row:

            # Digit crop karo
            digit = thresh[y:y+h, x:x+w]

            # Aspect ratio preserve karke resize karo
            scale = 20.0 / max(w, h)

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(
                digit,
                (new_w, new_h)
            )

            # 28x28 black canvas
            canvas = np.zeros(
                (28, 28),
                dtype=np.uint8
            )

            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2

            canvas[
                y_offset:y_offset + new_h,
                x_offset:x_offset + new_w
            ] = resized

            # Normalize image
            processed = canvas.astype("float32") / 255.0

            # CNN input shape
            processed = processed.reshape(
                1, 28, 28, 1
            )

            # Model prediction
            probabilities = model.predict(
                processed,
                verbose=0
            )[0]

            predicted_digit = int(
                np.argmax(probabilities)
            )

            confidence = float(
                np.max(probabilities) * 100
            )

            row_prediction += str(predicted_digit)

            all_results.append({
                "digit": predicted_digit,
                "confidence": round(confidence, 2)
            })

        final_rows.append(row_prediction)

    # Rows ko newline ke saath join karo
    final_number = "\n".join(final_rows)

    return {
        "prediction": final_number,
        "rows": final_rows,
        "digits": all_results
    }
