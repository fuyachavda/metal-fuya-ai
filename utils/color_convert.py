import cv2
import numpy as np
import os

def convert_gold_color(image_path, gold_type):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    rose_mask = cv2.inRange(hsv, (0, 50, 50), (15, 255, 255))

    if gold_type == 'yellow':
        hsv[..., 0] = np.where(yellow_mask > 0, hsv[..., 0] - 10, hsv[..., 0])
    elif gold_type == 'rose':
        hsv[..., 0] = np.where(rose_mask > 0, hsv[..., 0] + 10, hsv[..., 0])

    converted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out_path = image_path.replace(".png", "_converted.jpg")
    cv2.imwrite(out_path, converted)
    return out_path
