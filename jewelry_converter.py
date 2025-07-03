import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from io import BytesIO
import os
import concurrent.futures
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import color, exposure, filters

# Predefined target colors in LAB color space (OpenCV range)
TARGET_COLORS = {
    "yellow_gold": (224, 138, 183),  # LAB values for yellow gold (OpenCV range)
    "rose_gold": (157, 163, 143)      # LAB values for rose gold (OpenCV range)
}

# Set page configuration
st.set_page_config(
    page_title="Bulk Jewelry Metal Converter",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #D4AF37;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #ffd700, #ff69b4, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #D4AF37;
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .gold-button {
        background: linear-gradient(45deg, #D4AF37, #FFD700);
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        margin: 10px 0;
    }
    .gold-button:hover {
        background: linear-gradient(45deg, #B8860B, #D4AF37);
        color: white;
    }
    .conversion-card {
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
        background-color: #fffdf6;
    }
    .color-sample {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        display: inline-block;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .image-container {
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 15px;
        background-color: #fffdf6;
        margin: 15px 0;
        text-align: center;
    }
    .stApp {
        background-color: #fffaf0;
    }
    .processing-animation {
        text-align: center;
        padding: 20px;
    }
    .error {
        color: #ff4b4b;
        font-weight: bold;
        padding: 10px;
        border: 1px solid #ff4b4b;
        border-radius: 5px;
        background-color: #fff0f0;
    }
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .gallery-item {
        border: 1px solid #D4AF37;
        border-radius: 10px;
        padding: 15px;
        background-color: #fffdf6;
        text-align: center;
    }
    .progress-container {
        margin: 20px 0;
        padding: 15px;
        border: 1px solid #D4AF37;
        border-radius: 10px;
        background-color: #fffdf6;
    }
</style>
""", unsafe_allow_html=True)

def detect_metal_color(img_array):
    """Advanced metal color detection using histogram analysis"""
    # Convert to LAB color space
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Create metal mask
    metal_mask = create_metal_mask_combined(img_array)
    
    # If no metal detected, try alternative approach
    if np.sum(metal_mask) < 100:
        return "unknown"
    
    # Extract metal regions
    metal_a = a[metal_mask]
    metal_b = b[metal_mask]
    
    # Calculate histogram distributions
    a_hist = np.histogram(metal_a, bins=50, range=(0, 255))[0]
    b_hist = np.histogram(metal_b, bins=50, range=(0, 255))[0]
    
    # Calculate peak positions
    a_peak = np.argmax(a_hist) * 5
    b_peak = np.argmax(b_hist) * 5
    
    # Calculate distances to target colors
    dist_yellow = np.sqrt((a_peak - TARGET_COLORS["yellow_gold"][1])**2 + 
                         (b_peak - TARGET_COLORS["yellow_gold"][2])**2)
    
    dist_rose = np.sqrt((a_peak - TARGET_COLORS["rose_gold"][1])**2 + 
                       (b_peak - TARGET_COLORS["rose_gold"][2])**2)
    
    # Determine result with confidence
    if dist_yellow < 30 and dist_yellow < dist_rose:
        return "yellow_gold"
    elif dist_rose < 30 and dist_rose < dist_yellow:
        return "rose_gold"
    else:
        # Final fallback - HSV analysis
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        hsv_metal = hsv_img[metal_mask]
        h_hist = np.histogram(hsv_metal[:, 0], bins=50, range=(0, 180))[0]
        h_peak = np.argmax(h_hist) * 3.6
        
        if 15 <= h_peak <= 35:
            return "yellow_gold"
        elif h_peak <= 15 or h_peak >= 165:
            return "rose_gold"
        else:
            return "unknown"

def create_metal_mask_combined(img_array):
    """Create advanced mask for metal regions"""
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    
    # Create masks for different metal types
    # Yellow gold range (wider for complex designs)
    lower_yellow = np.array([10, 40, 50])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Rose gold range
    lower_rose1 = np.array([0, 40, 50])
    upper_rose1 = np.array([15, 255, 255])
    lower_rose2 = np.array([160, 40, 50])
    upper_rose2 = np.array([180, 255, 255])
    mask_rose1 = cv2.inRange(hsv, lower_rose1, upper_rose1)
    mask_rose2 = cv2.inRange(hsv, lower_rose2, upper_rose2)
    mask_rose = cv2.bitwise_or(mask_rose1, mask_rose2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_rose)
    
    # Apply morphological operations
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask > 0

def convert_metal_color(img_array, source_type):
    """Accurate color conversion with texture preservation"""
    # Convert to LAB
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    lab_float = lab_img.astype(np.float32)
    
    # Create metal mask
    mask = create_metal_mask_combined(img_array)
    
    # Get target color
    target_type = "yellow_gold" if source_type == "rose_gold" else "rose_gold"
    target_l, target_a, target_b = TARGET_COLORS[target_type]
    
    # Split channels
    l, a, b = cv2.split(lab_float)
    
    # Calculate mean color in metal regions
    mean_a = np.mean(a[mask])
    mean_b = np.mean(b[mask])
    
    # Apply color transformation only to metal regions
    a_new = np.where(mask, a - mean_a + target_a, a)
    b_new = np.where(mask, b - mean_b + target_b, b)
    
    # Merge channels
    converted_lab = cv2.merge([l, a_new, b_new])
    converted_lab = np.clip(converted_lab, 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    converted_bgr = cv2.cvtColor(converted_lab, cv2.COLOR_LAB2BGR)
    
    # Preserve original background
    result = img_array.copy()
    result[mask] = converted_bgr[mask]
    
    return result

def process_single_image(uploaded_file):
    """Process a single image file"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect metal color
        metal_type = detect_metal_color(img_array)
        
        # Convert if detection is successful
        if metal_type in ["yellow_gold", "rose_gold"]:
            converted_img = convert_metal_color(img_array, metal_type)
            converted_img_rgb = cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB)
            converted_pil = Image.fromarray(converted_img_rgb)
            
            return {
                "filename": uploaded_file.name,
                "original": image,
                "converted": converted_pil,
                "metal_type": metal_type,
                "status": "success"
            }
        else:
            return {
                "filename": uploaded_file.name,
                "original": image,
                "converted": image,  # Return original if conversion fails
                "metal_type": "unknown",
                "status": "failed"
            }
            
    except Exception as e:
        return {
            "filename": uploaded_file.name,
            "error": str(e),
            "status": "error"
        }

def create_zip_archive(results):
    """Create a ZIP file of converted images"""
    import zipfile
    from io import BytesIO
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            if result['status'] == 'success':
                img_byte_arr = BytesIO()
                result['converted'].save(img_byte_arr, format='JPEG', quality=95)
                zip_file.writestr(result['filename'], img_byte_arr.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    # App title
    st.markdown('<div class="header">💎 Bulk Jewelry Metal Converter</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <p>This advanced AI tool processes multiple jewelry images at once, detecting metal colors 
        and converting between rose gold and yellow gold while preserving original quality.</p>
        
        <p><b>Key Features:</b></p>
        <ul>
            <li>Bulk processing of multiple images</li>
            <li>Enhanced color detection algorithm</li>
            <li>Parallel processing for fast conversions</li>
            <li>Gallery view of results</li>
            <li>Bulk download as ZIP archive</li>
        </ul>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Upload multiple jewelry images (JPG/PNG)</li>
            <li>Click "Process Images" to start conversion</li>
            <li>View results in the gallery</li>
            <li>Download individual images or all as ZIP</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload jewelry images (JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Process button
    if st.button('Process Images', use_container_width=True, type="primary") and uploaded_files:
        st.session_state.processing = True
        st.session_state.processed_results = []
        st.session_state.progress = 0
        
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("Starting image processing...")
        
        # Process images with parallel execution
        total_files = len(uploaded_files)
        results = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_single_image, file): file for file in uploaded_files}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "filename": file.name,
                        "error": str(e),
                        "status": "error"
                    })
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                st.session_state.progress = progress
                progress_bar.progress(progress)
                status_text.info(f"Processing: {i+1}/{total_files} images ({progress}%)")
        
        st.session_state.processed_results = results
        st.session_state.processing = False
        status_text.success("Processing complete!")
        time.sleep(2)
        status_text.empty()
    
    # Display processing status
    if st.session_state.processing:
        st.markdown(f"""
        <div class="progress-container">
            <h3>Processing Images</h3>
            <p>Progress: {st.session_state.progress}%</p>
            <p>Please wait while your images are being processed...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results
    if st.session_state.processed_results:
        st.markdown('<div class="subheader">Conversion Results</div>', unsafe_allow_html=True)
        
        # Create ZIP archive
        zip_buffer = create_zip_archive(st.session_state.processed_results)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download All Images (ZIP)",
                data=zip_buffer,
                file_name="converted_jewelry.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        with col2:
            if st.button('Clear Results', use_container_width=True):
                st.session_state.processed_results = []
                st.experimental_rerun()
        
        # Gallery view
        st.markdown('<div class="gallery">', unsafe_allow_html=True)
        
        for result in st.session_state.processed_results:
            if result['status'] in ['success', 'failed']:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"<h4>Original: {result['filename']}</h4>", unsafe_allow_html=True)
                    st.image(result['original'], use_column_width=True)
                
                with col2:
                    if result['status'] == 'success':
                        st.markdown(f"<h4>Converted: {result['metal_type'].replace('_', ' ').title()} → {'Rose Gold' if 'yellow' in result['metal_type'] else 'Yellow Gold'}</h4>", 
                                   unsafe_allow_html=True)
                        st.image(result['converted'], use_column_width=True)
                        
                        # Download button for single image
                        buf = BytesIO()
                        result['converted'].save(buf, format="JPEG", quality=95)
                        st.download_button(
                            label=f"Download {result['filename']}",
                            data=buf.getvalue(),
                            file_name=f"converted_{result['filename']}",
                            mime="image/jpeg",
                            key=f"dl_{result['filename']}"
                        )
                    else:
                        st.markdown(f"<h4 style='color:red;'>Detection Failed: {result['filename']}</h4>", 
                                   unsafe_allow_html=True)
                        st.image(result['original'], use_column_width=True)
                        st.error("Could not detect metal color. Please try a clearer image.")
            
            elif result['status'] == 'error':
                st.error(f"Error processing {result['filename']}: {result['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #777; padding: 20px;">
        <p>Bulk Jewelry Metal Converter • Advanced AI Color Detection</p>
        <p>Process multiple images at once with professional results</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()