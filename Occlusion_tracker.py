import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

yolo_model = YOLO('yolov8m.pt')

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)

class_names = yolo_model.model.names

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (480, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    results = yolo_model(frame)
    detections = []
    
    for result in results:
        for det in result.boxes.data:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf > 0.4: 
                class_name = class_names.get(int(cls), "Unknown")
                detections.append(([x1, y1, x2, y2], conf, int(cls), class_name))
    
    tracked_objects = tracker.update_tracks([(d[0], d[1], d[2]) for d in detections], frame=frame)
    
    for track, (_, _, _, class_name) in zip(tracked_objects, detections):
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f'ID {track_id}: {class_name}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8 Occlusion Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()