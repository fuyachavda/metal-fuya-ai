import cv2
import numpy as np
import os
from PIL import Image

def segment_jewelry(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 10, 10), (35, 255, 255))  # yellow region mask (approx)

    masked = cv2.bitwise_and(img, img, mask=mask)
    output_path = image_path.replace(".png", "_seg.png")
    cv2.imwrite(output_path, masked)
    return output_path
