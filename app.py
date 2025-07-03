import cv2
import numpy as np
import time
import urllib.request
from sklearn.cluster import KMeans

# Predefined target colors in LAB color space
# (Calculated from provided example images)
TARGET_COLORS = {
    "yellow_gold": (87.95, 10.20, 54.72),  # LAB values for yellow gold
    "rose_gold": (61.40, 34.98, 14.63)     # LAB values for rose gold
}

def get_dominant_color(image, k=3):
    """Extract dominant color from image using K-Means clustering"""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color.astype(int)

def detect_metal_color(img):
    """Detect if jewelry is rose gold or yellow gold"""
    # Convert to LAB color space for accurate color analysis
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Calculate color distances to reference colors
    yellow_dist = np.sqrt(
        (a - TARGET_COLORS["yellow_gold"][1])**2 +
        (b - TARGET_COLORS["yellow_gold"][2])**2
    )
    
    rose_dist = np.sqrt(
        (a - TARGET_COLORS["rose_gold"][1])**2 +
        (b - TARGET_COLORS["rose_gold"][2])**2
    )
    
    # Determine dominant metal type
    yellow_pixels = np.sum(yellow_dist < rose_dist)
    rose_pixels = np.sum(rose_dist < yellow_dist)
    
    return "yellow_gold" if yellow_pixels > rose_pixels else "rose_gold"

def convert_metal_color(img, source_type):
    """Convert jewelry between rose gold and yellow gold"""
    # Convert to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Get target color based on conversion direction
    target_type = "yellow_gold" if source_type == "rose_gold" else "rose_gold"
    target_l, target_a, target_b = TARGET_COLORS[target_type]
    
    # Create masks for metal areas
    source_mask = create_metal_mask(img, source_type)
    
    # Apply color transformation only to metal regions
    a = np.where(source_mask, target_a + (a - np.mean(a)), a)
    b = np.where(source_mask, target_b + (b - np.mean(b)), b)
    
    # Merge channels and convert back to BGR
    converted_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(converted_lab, cv2.COLOR_LAB2BGR), source_mask

def create_metal_mask(img, metal_type):
    """Create mask for metal regions using adaptive thresholding"""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for different metal types
    if metal_type == "yellow_gold":
        lower = np.array([15, 40, 40])
        upper = np.array([30, 255, 255])
    else:  # rose_gold
        lower1 = np.array([0, 40, 40])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 40, 40])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask > 0
    
    return cv2.inRange(hsv, lower, upper) > 0

def process_image(image_path):
    """Main processing pipeline"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        # Try to download if it's a URL
        with urllib.request.urlopen(image_path) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not load image from source")
    
    print("Processing image...")
    time.sleep(2)  # Simulate processing delay
    
    # Step 1: Detect metal color
    print("Detecting metal color...")
    time.sleep(1)
    metal_type = detect_metal_color(img)
    print(f"Detected: {metal_type.replace('_', ' ').title()}")
    time.sleep(1)
    
    # Step 2: Convert metal color
    print("Converting metal color...")
    converted_img, metal_mask = convert_metal_color(img, metal_type)
    
    # Step 3: Preserve original background
    result = img.copy()
    result[metal_mask] = converted_img[metal_mask]
    
    return result

# Example usage
if __name__ == "__main__":
    input_image = "input_jewelry.jpg"  # Path to your image
    output_image = "converted_jewelry.jpg"
    
    # Process image
    result_img = process_image(input_image)
    
    # Save result
    cv2.imwrite(output_image, result_img)
    print(f"Conversion complete! Saved to {output_image}")