import cv2
import numpy as np


def segment_digits(image_path):
    # Image load karo
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image load nahi hui. Image path check karo.")

    # Grayscale mein convert karo
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black-and-white image banao
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Har digit ka contour find karo
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Chhote noise ko ignore karo
        if w < 5 or h < 10:
            continue

        # Digit crop karo
        digit = thresh[y:y+h, x:x+w]

        # Aspect ratio preserve karke 20x20 mein resize karo
        scale = 20 / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        digit = cv2.resize(digit, (new_w, new_h))

        # 28x28 canvas par center karo
        canvas = np.zeros((28, 28), dtype=np.uint8)

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        canvas[y_offset:y_offset+new_h,
               x_offset:x_offset+new_w] = digit

        digits.append((x, canvas))

    # Digits ko left-to-right arrange karo
    digits.sort(key=lambda item: item[0])

    return [digit for x, digit in digits]