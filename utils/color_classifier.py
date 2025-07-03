import cv2
import numpy as np

def detect_gold_color(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = img.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # remove black background

    avg = np.mean(pixels, axis=0)
    red, green, blue = avg

    if red > 180 and green > 100 and blue < 100:
        return "rose"
    elif red > 200 and green > 180 and blue < 100:
        return "yellow"
    else:
        return "unknown"
