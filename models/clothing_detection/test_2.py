import os
import cv2
import torch
from ultralytics import YOLO
import time
import datetime
import numpy as np
from collections import Counter

# Get the absolute path to the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model weights
model_path = os.path.join(base_dir, "weights", "best.pt")

# Load the YOLOv8 model
model = YOLO(model_path)

# Path to a single image or folder
image_path = os.path.join(base_dir, "khaki_pants.webp")  # Change if needed

# Create results directory if it doesn't exist
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Color detection function
def detect_color(image, x1, y1, x2, y2):
    # Extract region of interest (ROI)
    roi = image[y1:y2, x1:x2]
    
    # Resize ROI for faster processing if it's large
    height, width = roi.shape[:2]
    if height > 100 or width > 100:
        scale_factor = min(100 / height, 100 / width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        roi = cv2.resize(roi, (new_width, new_height))
    
    # Convert to HSV color space (better for color analysis)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'red': [(0, 70, 50), (10, 255, 255), (160, 70, 50), (180, 255, 255)],  # Red wraps around
        'orange': [(10, 70, 50), (25, 255, 255)],
        'yellow': [(25, 70, 50), (35, 255, 255)],
        'green': [(35, 70, 50), (85, 255, 255)],
        'blue': [(85, 70, 50), (130, 255, 255)],
        'purple': [(130, 70, 50), (160, 255, 255)],
        'brown': [(10, 50, 20), (30, 150, 100)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'gray': [(0, 0, 30), (180, 30, 150)],
        'white': [(0, 0, 150), (180, 30, 255)]
    }
    
    color_pixels = {}
    
    # Count pixels in each color range
    for color, ranges in color_ranges.items():
        # Some colors (like red) may have multiple ranges
        if len(ranges) == 4:  # Two ranges (e.g., red)
            mask1 = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
            mask2 = cv2.inRange(hsv_roi, np.array(ranges[2]), np.array(ranges[3]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # One range
            mask = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
        
        color_pixels[color] = cv2.countNonZero(mask)
    
    # Determine dominant color
    total_pixels = roi.shape[0] * roi.shape[1]
    for color in color_pixels:
        color_pixels[color] = color_pixels[color] / total_pixels
    
    dominant_color = max(color_pixels.items(), key=lambda x: x[1])
    
    # Only return a color if it represents a significant portion of the ROI
    if dominant_color[1] > 0.15:  # At least 15% of the pixels
        return dominant_color[0]
    else:
        return "unknown"

# Generate unique filename using timestamp
def get_unique_filename(prefix="result", extension=".jpg"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}{extension}"

# Run inference on a single image
if os.path.isfile(image_path):
    results = model(image_path)
    
    for result in results:
        boxes = result.boxes
        
        img = cv2.imread(image_path)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Detect dominant color
            color = detect_color(img, x1, y1, x2, y2)
            
            # Create label with class name, color and confidence
            label = f"{color} {class_name} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save with unique filename
        filename = get_unique_filename()
        save_path = os.path.join(results_dir, filename)
        cv2.imwrite(save_path, img)
        print(f"Detection complete. Result saved as '{save_path}'")

# Run inference on all images in a folder
elif os.path.isdir(image_path):
    for img_file in os.listdir(image_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_path, img_file)
            results = model(full_path)
            
            # Process and save results
            for result in results:
                boxes = result.boxes
                
                img = cv2.imread(full_path)
                base_name = os.path.splitext(img_file)[0]
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    # Detect dominant color
                    color = detect_color(img, x1, y1, x2, y2)
                    
                    # Create label with class name, color and confidence
                    label = f"{color} {class_name} {conf:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save with unique filename based on original name and timestamp
                extension = os.path.splitext(img_file)[1]
                filename = get_unique_filename(prefix=f"result_{base_name}", extension=extension)
                save_path = os.path.join(results_dir, filename)
                cv2.imwrite(save_path, img)
                print(f"Detection complete for {img_file}. Result saved as '{save_path}'")