import streamlit as st
import os, time
import cv2
import numpy as np
import pandas as pd
import psutil
from datetime import datetime
from yolo_wrapper import infer

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static_input")    # 用静态图片目录
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "inference_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(layout="wide")
st.title("YOLO Multi-Model Static Image Batch Inference Demo")

model_type = st.selectbox("Select Model", ["YOLOv8", "YOLOv5", "YOLOv3-Tiny"])

# 获取静态图片列表
img_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
log_data = []

if not img_list:
    st.warning("No images found in static_input/. Please add test images.")
else:
    # 添加图片选择器
    img_file = st.selectbox("Select image to test:", img_list)
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    results = infer(img, model_type)
    end_time = time.time()
    inf_time_ms = (end_time - start_time) * 1000
    fps = 1000 / inf_time_ms if inf_time_ms > 0 else 0
    cpu = psutil.cpu_percent(interval=0.01)
    mem = psutil.virtual_memory().percent
    timestamp = datetime.now().isoformat()

    # Draw detection boxes
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img_rgb, f"{det['class']} {det['confidence']:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    st.image(img_rgb, caption=f"{img_file} | Inference: {inf_time_ms:.1f} ms | FPS: {fps:.2f}", use_container_width=True)
    st.write("**Detection Results:**")
    for det in results:
        st.write(f"- {det['class']} ({det['confidence']*100:.1f}%) {det['bbox']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Inference Time (ms)", f"{inf_time_ms:.1f}")
    col2.metric("FPS", f"{fps:.2f}")
    col3.metric("CPU / Memory", f"{cpu}% / {mem}%")

    for det in results:
        log_data.append({
            "Time": timestamp,
            "Model": model_type,
            "Image": img_file,
            "Class": det["class"],
            "Confidence": det["confidence"],
            "BBox": det["bbox"],
            "Inference(ms)": inf_time_ms,
            "FPS": fps,
            "CPU(%)": cpu,
            "Mem(%)": mem
        })

    df = pd.DataFrame(log_data)
    if os.path.exists(LOG_FILE):
        df_old = pd.read_csv(LOG_FILE)
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

    st.subheader("Inference Time Trend (last 50 tests)")
    st.line_chart(df["Inference(ms)"].tail(50))

with st.expander("Download Detection Log (CSV)"):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download inference_log.csv", f, file_name="inference_log.csv", mime="text/csv")
    else:
        st.write("No log available")
