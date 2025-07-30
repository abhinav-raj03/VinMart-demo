import cv2
import torch
from ultralytics import YOLO
from tracker.tracker import DeepSort # Assuming this is your custom DeepSort import
from logger import log_detection
from utils import draw_box
import time
import os

# Ensure logs/ directory exists for the logger
os.makedirs("logs", exist_ok=True)

# Load YOLOv8 model
# Ensure 'models/yolo11s.pt' exists at this path
try:
    model = YOLO('models/yolo11m.pt')
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    print("Please ensure 'models/yolo11s.pt' exists and is a valid YOLOv8 model file.")
    exit()

# Initialize Deep SORT tracker
# Max IOU distance and max age are common parameters, adjust if needed by your tracker's constructor
try:
    tracker = DeepSort(max_iou_distance=0.7, max_age=30, n_init=3) # Common DeepSort constructor parameters
except TypeError:
    # If your custom tracker.tracker.DeepSort doesn't accept these args,
    # you might need to revert to `tracker = DeepSort()` or check its __init__ signature.
    print("Warning: DeepSort constructor arguments (max_iou_distance, max_age, n_init) might be incompatible. Trying without them.")
    tracker = DeepSort()


# Open video source
video_path = "data/bottle_1.mp4"
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at {video_path}. Please check the path.")
    print("Attempting to open webcam (index 0) instead...")
    cap = cv2.VideoCapture(0) # Fallback to webcam
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video source (webcam or file). Exiting.")
    exit()

# Setup video writer for output
output_video_path = 'output1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"🎥 Processing video. Output will be saved to '{output_video_path}'. Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_count += 1
    # print(f"Processing frame {frame_count}...")

    # Perform YOLO inference
    # results is a list of Results objects (one per image/frame in the batch)
    # [0] selects the results for the current frame
    yolo_results = model(frame, conf=0.4, verbose=False)[0] # verbose=False to suppress detailed output per frame

    # Prepare detections in the format expected by Deep SORT
    # Format: [([x, y, w, h], confidence, class_id), ...]
    deepsort_detections = []
    for rbox in yolo_results.boxes:
        # xyxy: [x1, y1, x2, y2] - top-left (x,y) and bottom-right (x,y)
        x1, y1, x2, y2 = map(int, rbox.xyxy[0])
        conf = float(rbox.conf[0])
        cls_id = int(rbox.cls[0])

        # Calculate width and height
        w = x2 - x1
        h = y2 - y1

        # DeepSORT expects bounding box as [x, y, w, h]
        bbox_xywh = [x1, y1, w, h]

        # Append as a tuple: (bounding_box_list, confidence, class_id)
        deepsort_detections.append((bbox_xywh, conf, cls_id))

    # Update tracks with the correctly formatted detections
    tracked_objects = tracker.update_tracks(deepsort_detections, frame=frame)

    # Draw bounding boxes and log detections for tracked objects
    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for track in tracked_objects:
        # Only draw and log confirmed tracks
        if not track.is_confirmed() or track.time_since_update > 1: # You can adjust time_since_update threshold
            continue

        track_id = track.track_id
        # to_ltrb() returns [left, top, right, bottom]
        l, t, r, b = track.to_ltrb()


        track_class_id = track.get_det_attr('class_id') if hasattr(track, 'get_det_attr') else cls_id # Fallback to last seen cls_id if available, or just 0

        center_x = int((l + r) / 2)
        center_y = int((t + b) / 2)

        # Log the detection (make sure your log_detection function can handle track_id as bottle_id)
        log_detection(track_id, track_class_id, current_timestamp, center_x, center_y)

        # Draw the bounding box
        draw_box(frame, (l, t, r, b), track_class_id, track_id)

    # Write the frame to the output video
    output.write(frame)

    # Display the frame
    cv2.imshow("Bottle Tracking", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting by user request.")
        break

# Release resources
output.release()
cap.release()
cv2.destroyAllWindows()
print("Video processing finished. Resources released.")
