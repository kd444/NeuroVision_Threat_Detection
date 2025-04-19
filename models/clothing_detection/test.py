import os
import cv2
import torch
from ultralytics import YOLO
import time
import datetime

# Get the absolute path to the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model weights
model_path = os.path.join(base_dir, "weights", "best.pt")

# Load the YOLOv8 model
model = YOLO(model_path)

# Path to a single image or folder
image_path = os.path.join(base_dir, "pants.jpg")  # Change if needed

# Create results directory if it doesn't exist
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

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
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
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
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save with unique filename based on original name and timestamp
                extension = os.path.splitext(img_file)[1]
                filename = get_unique_filename(prefix=f"result_{base_name}", extension=extension)
                save_path = os.path.join(results_dir, filename)
                cv2.imwrite(save_path, img)
                print(f"Detection complete for {img_file}. Result saved as '{save_path}'")