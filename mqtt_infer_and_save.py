import paho.mqtt.client as mqtt
import struct
import os
import time
import json
from collections import defaultdict
import cv2
from datetime import datetime
from yolo_wrapper import infer  # You should already have this wrapper

# MQTT settings
broker = "206.189.125.120"
port = 8888
username = "mouliu"
password = "lxy123123qq"
topic = "img"

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "live_input")  # ← MQTT saves raw image here
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "inference_log.csv")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Image chunk buffer
image_buffer = defaultdict(dict)
image_total_blocks = {}
image_received_count = {}

model_type = "YOLOv8"  # Default model

def run_inference(image_path, image_id):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start = time.time()
    results = infer(img, model_type)
    end = time.time()
    inf_time = (end - start) * 1000  # ms

    # Draw bounding boxes
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_rgb, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save result image
    out_path = os.path.join(RESULT_DIR, f"mqtt_{image_id}_det.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Save detection JSON
    result_json_path = os.path.join(LOG_DIR, f"mqtt_{image_id}.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Append log
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for det in results:
            f.write(f"{timestamp},{model_type},live_input,mqtt_{image_id}.jpg,{det['class']},{det['confidence']:.3f},{det['bbox']},{inf_time:.1f}\n")

    print(f"[✔] Inference complete for mqtt_{image_id} | {inf_time:.1f} ms")

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(topic)

def on_message(client, userdata, msg):
    payload = msg.payload
    if len(payload) < 8:
        return

    image_id, block_idx, total_blocks, length = struct.unpack('<HHHH', payload[:8])
    data = payload[8:]

    if len(data) != length:
        print(f"[Warning] Length mismatch: expected {length}, got {len(data)}")
        return

    image_buffer[image_id][block_idx] = data
    image_total_blocks[image_id] = total_blocks
    image_received_count[image_id] = image_received_count.get(image_id, 0) + 1

    if image_received_count[image_id] == total_blocks:
        print(f"[✓] Image {image_id} fully received. Saving and running inference...")
        all_data = b''.join(image_buffer[image_id][i] for i in range(total_blocks))
        img_path = os.path.join(INPUT_DIR, f"mqtt_{image_id}.jpg")
        with open(img_path, "wb") as f:
            f.write(all_data)

        run_inference(img_path, image_id)

        # Cleanup
        del image_buffer[image_id]
        del image_total_blocks[image_id]
        del image_received_count[image_id]

# Start MQTT client
client = mqtt.Client()
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker, port, 60)

print("[MQTT] Listening for incoming image blocks...")
client.loop_forever()
