from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # path to your YOLOv8 .pt file

# Load an image (update with your test image)
image_path = "test2.jpeg"
results = model(image_path, show=True)  # show=True opens a window with results

# Save results
results[0].save(filename="result2.jpg")
