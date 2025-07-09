import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO as YOLOv8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== YOLOv3-Tiny ==========
YOLOV3_CFG = os.path.join(BASE_DIR, "models", "v3tiny", "yolov3-tiny.cfg")
YOLOV3_WEIGHTS = os.path.join(BASE_DIR, "models", "v3tiny", "yolov3-tiny.weights")
YOLOV3_NAMES = os.path.join(BASE_DIR, "models", "v3tiny", "coco.names")

net_v3 = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHTS)
net_v3.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_v3.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(YOLOV3_NAMES, "r") as f:
    classes_v3 = [line.strip() for line in f.readlines()]

def infer_v3_tiny(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net_v3.setInput(blob)
    out_layers = net_v3.getUnconnectedOutLayersNames()
    layer_outputs = net_v3.forward(out_layers)
    results = []
    for output in layer_outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = det[0:4] * np.array([width, height, width, height])
                cx, cy, w, h = box.astype("int")
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                results.append({
                    "class": classes_v3[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": (x, y, x + int(w), y + int(h))
                })
    return results

# ========== YOLOv5 ==========
YOLOV5_PT = os.path.join(BASE_DIR, "models", "v5", "yolov5s.pt")
model_v5 = torch.hub.load("ultralytics/yolov5", "custom", path=YOLOV5_PT)

def infer_v5(image):
    results = model_v5(image)
    df = results.pandas().xyxy[0]
    return [
        {
            "class": row["name"],
            "confidence": round(row["confidence"], 3),
            "bbox": (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
        } for _, row in df.iterrows()
    ]

# ========== YOLOv8 ==========
YOLOV8_PT = os.path.join(BASE_DIR, "models", "v8", "yolov8n.pt")
model_v8 = YOLOv8(YOLOV8_PT)

def infer_v8(image):
    results = model_v8(image)
    dets = results[0].boxes.data.cpu().numpy()
    names = results[0].names
    return [
        {
            "class": names[int(cls)],
            "confidence": round(conf, 3),
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        } for x1, y1, x2, y2, conf, cls in dets
    ]

def infer(image, model_type):
    if model_type == "YOLOv3-Tiny":
        return infer_v3_tiny(image)
    elif model_type == "YOLOv5":
        return infer_v5(image)
    elif model_type == "YOLOv8":
        return infer_v8(image)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
