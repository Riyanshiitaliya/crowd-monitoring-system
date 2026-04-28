import cv2
from ultralytics import YOLO
import pyautogui

# Load YOLOv8 model (change model size for more accuracy)
model = YOLO("yolov8n.pt")  # Try yolov8l.pt for best accuracy, but slower

# Check for GPU
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
print("✅ Using device:", device)

# Load video
video_path = r"D:\detection\videos\crowd 2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

# Get screen size
screen_width, screen_height = pyautogui.size()

# Track total unique person IDs
unique_ids = set()

# Changeable detection settings
CONF_THRESHOLD = 0.2  # Lower = detect more, but can give false positives
IMG_SIZE = 960  # Increase for more detail (640 default)

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video.")
        break

    # Run tracking with custom conf & img size
    results = model.track(
        frame,
        persist=True,
        classes=[0],  # 0 = person
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device
    )

    if results and results[0].boxes.id is not None:
        for box in results[0].boxes:
            if box.id is not None:
                person_id = int(box.id[0])
                unique_ids.add(person_id)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Auto resize to fit screen
    h, w = annotated_frame.shape[:2]

    scale_w = (screen_width - 100) / w
    scale_h = (screen_height - 150) / h
    scale_factor = min(scale_w, scale_h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))

    # Show video


    cv2.imshow("YOLOv8 - People Counting", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ Final result
print(f"\n👥 Total People Detected in Video: {len(unique_ids)}")
