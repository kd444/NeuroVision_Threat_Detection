from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import torch
import numpy as np
import sys
import os
import time
from collections import defaultdict
from sklearn.cluster import KMeans

# Add YOLOv7 repository to the system path (only needed if using YOLOv7 for weapon detection/person fallback)
sys.path.insert(0, os.path.join(os.getcwd(), 'yolov7'))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

# Initialize FastAPI app
app = FastAPI()

# -------------------------------------------------------------------
# Updated HSV color ranges for improved color detection
COLOR_NAMES = {
    'red':   ([0, 70, 50], [10, 255, 255]),      # Red range 1
    'red2':  ([170, 70, 50], [180, 255, 255]),     # Red range 2
    'orange': ([11, 70, 50], [25, 255, 255]),
    'yellow': ([26, 70, 50], [35, 255, 255]),
    'green': ([36, 70, 50], [85, 255, 255]),
    'blue': ([86, 70, 50], [130, 255, 255]),
    'purple': ([131, 70, 50], [160, 255, 255]),
    'pink': ([161, 50, 50], [169, 255, 255]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'black': ([0, 0, 0], [180, 255, 30]),
    'gray': ([0, 0, 31], [180, 30, 199]),
    'brown': ([0, 50, 20], [20, 200, 100])
}

# DeepFashion2 class names (adjust or extend based on your model's available classes)
DEEPFASHION2_CLASSES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress'
]

# Initialize device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------
# Load weapon detection model (YOLOv7)
try:
    weapon_model = attempt_load('yolov7-weapon-detection.pt', map_location=device)
    weapon_model.eval()
    print("Weapon detection model loaded successfully")
except Exception as e:
    print(f"Error loading weapon model: {e}")
    raise

# Import YOLOv8 from Ultralytics (for person and fashion detection)
try:
    from ultralytics import YOLO
    print("Ultralytics YOLO imported successfully")
    using_ultralytics = True
except ImportError:
    print("Ultralytics not available, falling back to YOLOv7 for person detection")
    using_ultralytics = False

# Load person detection model and DeepFashion2 model
if using_ultralytics:
    try:
        # Person detection model using YOLOv8
        person_model = YOLO("yolov8n.pt")
        print("Using Ultralytics YOLOv8 for person detection")
        # Updated DeepFashion2 clothing detection model (with segmentation support)
        # (You can alternatively use: YOLO("keremberke/yolov8m-deepfashion2") )
        fashion_model = YOLO("keremberke/yolov8m-deepfashion2")
        print("DeepFashion2 YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading Ultralytics models: {e}")
        raise
else:
    try:
        # Fallback to YOLOv7 for person detection only
        person_model = attempt_load('yolov7.pt', map_location=device)
        person_model.eval()
        print("No DeepFashion2 model available without Ultralytics")
    except Exception as e:
        print(f"Error loading person model: {e}")
        raise

# -------------------------------------------------------------------
# Person tracker dictionary for maintaining state across frames
person_tracker = defaultdict(lambda: {
    'track_id': 0,
    'bbox': None,
    'clothing_color': None,
    'clothing_colors_history': [],
    'clothing_items': [],
    'last_seen': 0
})
next_track_id = 1  # For generating unique tracking IDs

# Initialize the webcam capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        raise Exception("Webcam not available")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    raise

# Statistics for display on the UI
stats = {
    'weapons_detected': 0,
    'persons_tracked': 0,
    'clothing_items_detected': 0,
    'processing_time': 0,
    'alerts': []
}

# -------------------------------------------------------------------
# Improved dominant color extraction function
def extract_dominant_color(frame, mask=None, bbox=None, top_portion=0.6):
    """
    Extract the dominant color from an image region with improved color detection.
    If a mask is provided, only the masked region is considered. Otherwise, if bbox is provided,
    a larger portion of the body (skipping the head) is analyzed.
    """
    if mask is not None:
        # Use the mask to extract relevant pixels
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        non_zero_pixels = roi[np.where(mask > 0)]
        if non_zero_pixels.size == 0:
            return None, None
        pixels = non_zero_pixels.reshape(-1, 3).astype(np.float32)
    elif bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        height = y2 - y1
        # Skip head/neck area and include more of the upper/middle body
        clothing_y1 = y1 + int(height * 0.15)
        clothing_y2 = y1 + int(height * top_portion)
        clothing_y1 = max(0, clothing_y1)
        clothing_y2 = min(frame.shape[0], clothing_y2)
        if clothing_y2 <= clothing_y1 or x2 <= x1:
            return None, None
        clothing_roi = frame[clothing_y1:clothing_y2, x1:x2]
        if clothing_roi.size == 0:
            return None, None
        # Apply slight color enhancement for more vivid detection
        clothing_roi = cv2.convertScaleAbs(clothing_roi, alpha=1.1, beta=5)
        pixels = clothing_roi.reshape(-1, 3).astype(np.float32)
    else:
        return None, None

    if len(pixels) < 10:
        return None, None

    # Convert pixels to HSV for color analysis
    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8),
                              cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Filter out very dark or very light pixels (to remove background/noise)
    valid_pixels = []
    for pixel in hsv_pixels:
        h, s, v = pixel
        if s > 30 and 20 < v < 240:
            valid_pixels.append(pixel)
    if not valid_pixels:
        return "gray", (np.array([128, 128, 128]), 100)

    hsv_pixels = np.array(valid_pixels)

    # Cluster pixels using K-means; use up to 3 clusters
    k = min(3, len(hsv_pixels))
    if k < 1:
        return None, None
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(hsv_pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    dominant_hsv = sorted_colors[0]
    color_name = identify_color(dominant_hsv)
    # Convert dominant color back to BGR (for visualization)
    dominant_bgr = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    percentage = (sorted_counts[0] / sum(sorted_counts)) * 100

    return color_name, (dominant_bgr, percentage)

# -------------------------------------------------------------------
# Improved color identification function
def identify_color(hsv_color):
    """
    Identify the color name from HSV values using improved thresholds.
    Special cases for low-saturation (grayscale), red (wrap-around) and brown.
    """
    h, s, v = hsv_color
    # Handle grayscale colors based on saturation and brightness
    if s < 30:
        if v < 30:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"
    h = h % 180  # Ensure hue is within 0-180 range for OpenCV
    # Special handling for red (covers wrap-around)
    if (h <= 10 or h >= 170) and s > 70 and v > 50:
        return "red"
    # Special case for brown (low saturation reddish/orange)
    if 0 <= h <= 20 and 50 <= s <= 200 and 20 <= v <= 100:
        return "brown"
    # Check each defined color range (skip special cases already handled)
    for color_name, (lower, upper) in COLOR_NAMES.items():
        if color_name in ['red', 'red2', 'brown']:
            continue
        if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
            return color_name
    return "unknown"

# -------------------------------------------------------------------
# Tracking and assignment functions remain largely unchanged
def assign_track_id(current_bbox, max_distance=50):
    """
    Assign track IDs to detected persons based on bounding box proximity.
    """
    global next_track_id, person_tracker
    current_time = time.time()
    # Purge tracks that have not been seen for over 5 seconds
    to_remove = []
    for track_id in list(person_tracker.keys()):
        if current_time - person_tracker[track_id]['last_seen'] > 5:
            to_remove.append(track_id)
    for track_id in to_remove:
        del person_tracker[track_id]
    cx = (current_bbox[0] + current_bbox[2]) / 2
    cy = (current_bbox[1] + current_bbox[3]) / 2
    best_match_id = None
    min_distance = float('inf')
    for track_id, data in person_tracker.items():
        if data['bbox'] is not None:
            tx = (data['bbox'][0] + data['bbox'][2]) / 2
            ty = (data['bbox'][1] + data['bbox'][3]) / 2
            distance = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                best_match_id = track_id
    if best_match_id is not None:
        return best_match_id
    else:
        new_id = next_track_id
        next_track_id += 1
        person_tracker[new_id] = {
            'track_id': new_id,
            'bbox': current_bbox,
            'clothing_color': None,
            'clothing_colors_history': [],
            'clothing_items': [],
            'last_seen': current_time
        }
        return new_id

def update_clothing_color(track_id, color_name):
    """
    Update the clothing color for a tracked person using a voting mechanism.
    """
    if track_id not in person_tracker:
        return
    person_tracker[track_id]['clothing_colors_history'].append(color_name)
    if len(person_tracker[track_id]['clothing_colors_history']) > 10:
        person_tracker[track_id]['clothing_colors_history'].pop(0)
    color_counts = {}
    for color in person_tracker[track_id]['clothing_colors_history']:
        color_counts[color] = color_counts.get(color, 0) + 1
    if color_counts:
        most_common_color = max(color_counts.items(), key=lambda x: x[1])[0]
        person_tracker[track_id]['clothing_color'] = most_common_color

def assign_clothing_to_person(person_boxes, clothing_results):
    """
    Match clothing detections to persons based on Intersection over Union (IoU).
    """
    assignments = {}
    for person_idx, person_box in enumerate(person_boxes):
        px1, py1, px2, py2 = person_box
        p_area = (px2 - px1) * (py2 - py1)
        assignments[person_idx] = []
        for clothing_idx, result in enumerate(clothing_results):
            if not using_ultralytics:
                continue
            cx1, cy1, cx2, cy2 = result.boxes.xyxy[0].tolist()
            ix1 = max(px1, cx1)
            iy1 = max(py1, cy1)
            ix2 = min(px2, cx2)
            iy2 = min(py2, cy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            i_area = (ix2 - ix1) * (iy2 - iy1)
            c_area = (cx2 - cx1) * (cy2 - cy1)
            iou = i_area / float(p_area + c_area - i_area)
            if iou > 0.1:
                assignments[person_idx].append((clothing_idx, iou))
    for person_idx in assignments:
        assignments[person_idx].sort(key=lambda x: x[1], reverse=True)
    return assignments

# -------------------------------------------------------------------
# Main processing loop for video frames
def process_frame():
    """
    Process video frames, applying person detection, clothing detection/color analysis,
    and weapon detection. The results are overlaid on the frame.
    """
    global stats
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        start_time = time.time()
        stats['alerts'] = []

        # Preprocess frame for detection models
        img = letterbox(frame, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img_person = img.clone()
        img_weapon = img.clone()

        # Run person detection
        person_boxes = []
        if using_ultralytics:
            person_results = person_model(frame, classes=[0], conf=0.4)  # Class 0: person (COCO)
            for result in person_results:
                for box in result.boxes:
                    if box.cls[0] == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_boxes.append([x1, y1, x2, y2])
        else:
            with torch.no_grad():
                person_results = person_model(img_person)[0]
            person_results = non_max_suppression(person_results, conf_thres=0.4, iou_thres=0.45, classes=[0])
            for i, det in enumerate(person_results):
                if len(det):
                    det[:, :4] = scale_coords(img_person.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        bbox = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        person_boxes.append(bbox)

        # Run DeepFashion2 clothing detection if using Ultralytics
        clothing_detections = []
        clothing_count = 0
        if using_ultralytics:
            clothing_results = fashion_model(frame)
            for result in clothing_results:
                for i, box in enumerate(result.boxes):
                    clothing_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    clothing_box = [x1, y1, x2, y2]
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if cls_id < len(DEEPFASHION2_CLASSES):
                        cls_name = DEEPFASHION2_CLASSES[cls_id]
                    else:
                        cls_name = f"clothing_{cls_id}"
                    mask = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        if len(result.masks.data) > i:
                            mask = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                    color_name, color_info = extract_dominant_color(frame, mask=mask, bbox=clothing_box)
                    clothing_detections.append({
                        'box': clothing_box,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'color': color_name,
                        'mask': mask,
                        'color_info': color_info
                    })
            stats['clothing_items_detected'] = clothing_count

        stats['persons_tracked'] = len(person_boxes)
        clothing_assignments = assign_clothing_to_person(
            person_boxes, clothing_results if using_ultralytics else []
        )

        # Process person tracks and update clothing color history
        for idx, bbox in enumerate(person_boxes):
            track_id = assign_track_id(bbox)
            color_name, color_info = extract_dominant_color(frame, bbox=bbox)
            if color_name:
                update_clothing_color(track_id, color_name)
            person_tracker[track_id]['bbox'] = bbox
            person_tracker[track_id]['last_seen'] = time.time()
            clothing_color = person_tracker[track_id]['clothing_color'] or "unknown"

            # Associate clothing detections with this person based on IoU
            if idx in clothing_assignments:
                person_tracker[track_id]['clothing_items'] = []
                for clothing_idx, iou in clothing_assignments[idx]:
                    if clothing_idx < len(clothing_detections):
                        item = clothing_detections[clothing_idx]
                        person_tracker[track_id]['clothing_items'].append({
                            'class_name': item['class_name'],
                            'color': item['color'] or clothing_color,
                            'confidence': item['confidence']
                        })

            label_parts = [f"Person #{track_id}"]
            if clothing_color != "unknown":
                label_parts.append(f"({clothing_color})")
            if person_tracker[track_id]['clothing_items']:
                items_text = ", ".join([
                    f"{item['color']} {item['class_name']}" 
                    for item in person_tracker[track_id]['clothing_items'][:2]
                ])
                if items_text:
                    label_parts.append(f"- {items_text}")
            label = " ".join(label_parts)
            plot_one_box([bbox[0], bbox[1], bbox[2], bbox[3]], display_frame,
                         label=label, color=(0, 255, 0), line_thickness=2)
            # Draw color patch if available
            if color_info:
                dominant_color, _ = color_info
                color_patch_size = 30
                x1, y1 = int(bbox[0]), int(bbox[1]) - color_patch_size - 5
                if y1 < 0:
                    y1 = int(bbox[3]) + 5
                cv2.rectangle(display_frame, (x1, y1),
                              (x1 + color_patch_size, y1 + color_patch_size),
                              (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])), -1)

        # Visualize clothing detections
        if using_ultralytics:
            for item in clothing_detections:
                box = item['box']
                item_color = item['color'] or "unknown"
                class_name = item['class_name']
                if item['mask'] is not None:
                    mask_color = np.array([100, 100, 255]) if item_color == "unknown" else (
                        item['color_info'][0] if item['color_info'] else np.array([100, 100, 255])
                    )
                    colored_mask = np.zeros_like(frame)
                    colored_mask[item['mask'] > 0] = mask_color
                    alpha = 0.4
                    mask_area = (item['mask'] > 0)
                    display_frame[mask_area] = cv2.addWeighted(
                        display_frame[mask_area], 1 - alpha, colored_mask[mask_area], alpha, 0
                    )
                label = f"{class_name}: {item_color}"
                cv2.rectangle(display_frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 255), 2)
                cv2.putText(display_frame, label, (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Run weapon detection
        with torch.no_grad():
            weapon_results = weapon_model(img_weapon)[0]
        weapon_results = non_max_suppression(weapon_results, conf_thres=0.4, iou_thres=0.45)
        weapon_count = 0
        for i, det in enumerate(weapon_results):
            if len(det):
                weapon_count += len(det)
                det[:, :4] = scale_coords(img_weapon.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    weapon_type = weapon_model.names[int(cls)]
                    label = f"{weapon_type} {conf:.2f}"
                    plot_one_box(xyxy, display_frame, label=label, color=(0, 0, 255), line_thickness=3)
                    weapon_center_x = (xyxy[0] + xyxy[2]) / 2
                    weapon_center_y = (xyxy[1] + xyxy[3]) / 2
                    min_distance = float('inf')
                    closest_person_id = None
                    closest_person_clothing = []
                    for track_id, data in person_tracker.items():
                        if data['bbox'] is not None:
                            person_center_x = (data['bbox'][0] + data['bbox'][2]) / 2
                            person_center_y = (data['bbox'][1] + data['bbox'][3]) / 2
                            distance = np.sqrt((weapon_center_x - person_center_x) ** 2 +
                                               (weapon_center_y - person_center_y) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_person_id = track_id
                                closest_person_clothing = data['clothing_items']
                    if closest_person_id is not None and min_distance < 200:
                        person_color = person_tracker[closest_person_id]['clothing_color'] or "unknown"
                        clothing_desc = ""
                        if closest_person_clothing:
                            clothing_items = [f"{item['color']} {item['class_name']}" for item in closest_person_clothing[:2]]
                            clothing_desc = f" wearing {', '.join(clothing_items)}"
                        alert_text = f"ALERT: Person #{closest_person_id} ({person_color}){clothing_desc} with {weapon_type}"
                        stats['alerts'].append({
                            'person_id': closest_person_id,
                            'color': person_color,
                            'clothing': closest_person_clothing,
                            'weapon': weapon_type,
                            'confidence': float(conf)
                        })
                        cv2.putText(display_frame, alert_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stats['weapons_detected'] = weapon_count

        # Overlay statistics on the frame
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, display_frame.shape[0] - 130),
                      (300, display_frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(overlay, f"Persons tracked: {stats['persons_tracked']}",
                    (20, display_frame.shape[0] - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Clothing items: {stats['clothing_items_detected']}",
                    (20, display_frame.shape[0] - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Weapons detected: {stats['weapons_detected']}",
                    (20, display_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if stats['weapons_detected'] > 0 else (255, 255, 255), 1)
        processing_time = time.time() - start_time
        stats['processing_time'] = processing_time
        cv2.putText(overlay, f"Processing time: {processing_time:.3f}s",
                    (20, display_frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        if stats['alerts']:
            alert_y = 60
            for alert in stats['alerts']:
                clothing_desc = ""
                if 'clothing' in alert and alert['clothing']:
                    clothing_items = [f"{item['color']} {item['class_name']}" for item in alert['clothing'][:1]]
                    clothing_desc = f" with {clothing_items[0]}"
                alert_text = f"THREAT: Person #{alert['person_id']} ({alert['color']}){clothing_desc} - {alert['weapon']}"
                cv2.putText(display_frame, alert_text, (10, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_y += 25
        _, jpeg = cv2.imencode('.jpg', display_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# -------------------------------------------------------------------
# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
    <head>
      <title>Advanced Clothing and Weapon Detection System</title>
      <style>
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          margin: 0;
          padding: 20px;
          text-align: center;
          background-color: #f8f9fa;
        }
        h1 { color: #dc3545; margin-bottom: 30px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .video-container { margin-top: 20px; background-color: #000; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .info-container { display: flex; justify-content: space-between; margin-top: 30px; }
        .info-box { flex: 1; margin: 0 10px; text-align: left; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .alert-box { margin-top: 20px; text-align: left; background-color: #fff; padding: 20px; border-radius: 8px; border-left: 5px solid #dc3545; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h2 { color: #343a40; font-size: 1.4rem; margin-top: 0; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; }
        ul { padding-left: 20px; }
        li { margin-bottom: 8px; }
        .highlight { font-weight: bold; color: #dc3545; }
        .footer { margin-top: 40px; color: #6c757d; font-size: 0.9rem; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>🚨 Advanced Clothing & Threat Detection System</h1>
        <div class="video-container">
            <img src="/video" width="100%" />
        </div>
        <div class="info-container">
          <div class="info-box">
            <h2>System Capabilities</h2>
            <ul>
              <li><span class="highlight">Weapon Detection:</span> Identifies guns and other weapons using YOLOv7</li>
              <li><span class="highlight">Clothing Recognition:</span> Detects specific clothing items using DeepFashion2</li>
              <li><span class="highlight">Person Tracking:</span> Tracks individuals with persistent IDs</li>
              <li><span class="highlight">Clothing Color Analysis:</span> Identifies the dominant color of each clothing item</li>
            </ul>
          </div>
          <div class="info-box">
            <h2>DeepFashion2 Categories</h2>
            <ul>
              <li>Short/long sleeve tops</li>
              <li>Short/long sleeve outerwear</li>
              <li>Vests and slings</li>
              <li>Shorts, trousers, and skirts</li>
              <li>Various types of dresses</li>
            </ul>
          </div>
        </div>
        <div class="alert-box">
          <h2>Threat Monitoring</h2>
          <p>The system monitors for weapons and provides detailed descriptions of individuals, including clothing type and color. Each alert includes:</p>
          <ul>
            <li>Person ID and tracking information</li>
            <li>Detailed clothing description (item type and color)</li>
            <li>Weapon type and proximity alerts</li>
            <li>Real-time segmentation of clothing items</li>
          </ul>
        </div>
        <div class="footer">
          <p>Advanced Threat Detection System | YOLOv8-DeepFashion2 Integration</p>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video")
def video_feed():
    return StreamingResponse(process_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
