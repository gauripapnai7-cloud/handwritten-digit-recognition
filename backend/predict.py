
import numpy as np
import cv2


def predict_multiple_digits(image, model):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 5 or h < 10:
            continue

        digit_boxes.append((x, y, w, h))

    if not digit_boxes:
        raise ValueError("Image mein koi digit nahi mila.")

    digit_boxes.sort(key=lambda box: box[1])

    rows = []

    for box in digit_boxes:
        x, y, w, h = box
        center_y = y + h / 2

        matched_row = None

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

    for row in rows:
        row.sort(key=lambda box: box[0])

    rows.sort(
        key=lambda row: min(box[1] for box in row)
    )

    processed_digits = []
    digit_positions = []

    for row_index, row in enumerate(rows):

        for x, y, w, h in row:

            digit = thresh[y:y+h, x:x+w]

            scale = 20.0 / max(w, h)

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(
                digit,
                (new_w, new_h)
            )

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

            processed_digits.append(
                canvas.astype("float32") / 255.0
            )

            digit_positions.append(row_index)

    batch = np.array(
        processed_digits,
        dtype="float32"
    ).reshape(-1, 28, 28, 1)

    predictions = model.predict(
        batch,
        verbose=0
    )

    all_results = []
    final_rows = ["" for _ in rows]

    for index, probabilities in enumerate(predictions):

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)

        row_index = digit_positions[index]

        final_rows[row_index] += str(predicted_digit)

        all_results.append({
            "digit": predicted_digit,
            "confidence": round(confidence, 2)
        })

    final_number = "\n".join(final_rows)

    return {
        "prediction": final_number,
        "rows": final_rows,
        "digits": all_results
    }
