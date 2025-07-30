# utils.py

import cv2

def draw_box(frame, box, class_id, track_id):
    l, t, r, b = map(int, box)
    color = (0, 255, 0)
    label = f"ID:{track_id} Class:{class_id}"
    cv2.rectangle(frame, (l, t), (r, b), color, 2)
    cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
