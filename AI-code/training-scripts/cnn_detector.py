import os
import torch
import numpy as np
from ultralytics import YOLO

class CNNDetector:
    def __init__(self, weights="models/target.pt", fallback="yolov8n.pt", conf=0.4):
        """
        weights: path to  custom YOLO weights
        fallback: fallback YOLO model
        conf: confidence threshold
        """
        if os.path.exists(weights):
            self.model = YOLO(weights)
            print(f"[CNNDetector] Loaded custom YOLO weights: {weights}")
        else:
            self.model = YOLO(fallback)
            print(f"[CNNDetector] Custom weights not found, using fallback: {fallback}")
        self.conf = conf

    def detect(self, frame_bgr):
        """
        frame_bgr: numpy image (OpenCV format, BGR)
        Returns: (cx, cy, r) of largest detected object, or None if no detection
        """
      
        frame_rgb = frame_bgr[:, :, ::-1]

        # Run inference
        results = self.model.predict(frame_rgb, conf=self.conf, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
       
        areas = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            areas.append((i, (x2-x1)*(y2-y1)))
        idx = max(areas, key=lambda t: t[1])[0]

        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].tolist())
        cx, cy = (x1+x2)//2, (y1+y2)//2
        r = int(0.5*max(x2-x1, y2-y1))  # radius proxy from box size

        return (cx, cy, r)
