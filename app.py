import streamlit as st
import os, time, json
import cv2
import numpy as np
import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from yolo_wrapper import infer

# ==== Paths ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static_input")
LIVE_DIR = os.path.join(BASE_DIR, "live_input")
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "inference_log.csv")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==== UI ====
st.set_page_config(layout="wide")
st.title("YOLO Multi-Model Image Inference Dashboard")

model_type = st.selectbox("Select Model", ["YOLOv8", "YOLOv5", "YOLOv3-Tiny"])
source_folder = st.selectbox("Select Image Source", ["static_input", "live_input"])
IMAGE_DIR = STATIC_DIR if source_folder == "static_input" else LIVE_DIR

# ==== BMP Fetch Button ====
RAW_URL = "http://206.189.125.120/view.php?res=raw"

def fetch_and_process_bmps():
    resp = requests.get(RAW_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    bmp_links = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.lower().endswith(".bmp"):
            bmp_links.append(urljoin(RAW_URL, href))

    st.info(f"Found {len(bmp_links)} BMP files.")

    for url in bmp_links:
        bmp_name = os.path.basename(url)
        bmp_path = os.path.join(LIVE_DIR, bmp_name)
        jpg_name = bmp_name.replace(".bmp", ".jpg")
        jpg_path = os.path.join(LIVE_DIR, jpg_name)

        if os.path.exists(jpg_path):
            st.write(f"✅ Already processed: {jpg_name}")
            continue

        try:
            bmp_data = requests.get(url).content
            with open(bmp_path, "wb") as f:
                f.write(bmp_data)

            img = cv2.imread(bmp_path)
            if img is None:
                st.error(f"Failed to read: {bmp_name}")
                continue
            cv2.imwrite(jpg_path, img)
            os.remove(bmp_path)

            results = infer(img, model_type)
            timestamp = datetime.now().isoformat()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for det in results:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            result_path = os.path.join(RESULT_DIR, jpg_name.replace(".jpg", "_det.jpg"))
            cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            json_path = os.path.join(LOG_DIR, jpg_name.replace(".jpg", ".json"))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                for det in results:
                    f.write(f"{timestamp},{model_type},raw,{jpg_name},{det['class']},{det['confidence']:.3f},{det['bbox']},NA\n")

            st.success(f"✓ Processed {jpg_name}")

        except Exception as e:
            st.error(f"Failed to process {url}: {e}")

st.markdown("### 📥 Download and Process BMP Images from Server")
if st.button("Fetch from http://206.189.125.120/view.php?res=raw"):
    fetch_and_process_bmps()

# ==== Inference on selected image ====
img_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
log_data = []

if not img_list:
    st.warning(f"No images found in {source_folder}/. Please add or fetch images.")
else:
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

    # Layout: image left, data right
    col_img, col_data = st.columns([2, 1])

    with col_img:
        for det in results:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        st.image(img_rgb, caption=f"{img_file}", width=512)

    with col_data:
        st.markdown("#### Detection Results")
        for det in results:
            st.write(f"- **{det['class']}** ({det['confidence']*100:.1f}%) {det['bbox']}")

        st.markdown("#### Performance")
        st.metric("Inference Time (ms)", f"{inf_time_ms:.1f}")
        st.metric("FPS", f"{fps:.2f}")
        st.metric("CPU / Memory", f"{cpu}% / {mem}%")

    for det in results:
        log_data.append({
            "Time": timestamp,
            "Model": model_type,
            "Source": source_folder,
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

    st.subheader("📊 Inference Time Trend (last 50 tests)")
    st.line_chart(df["Inference(ms)"].tail(50))

with st.expander("📁 Download Inference Log (CSV)"):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download inference_log.csv", f, file_name="inference_log.csv", mime="text/csv")
    else:
        st.write("No log available")
# ==== 清除推理日志 ====
st.markdown("### 🧹 Clear Inference History")
if st.button("Clear Inference Log"):
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Inference log cleared.")
    except Exception as e:
        st.error(f"Failed to delete log: {e}")

# ==== 平均推理时间图表 ====
st.markdown("### 📊 Average Inference Time per Image (last 50)")
if os.path.exists(LOG_FILE):
    df_log = pd.read_csv(LOG_FILE)
    if not df_log.empty:
        df_img_avg = df_log.groupby("Image")["Inference(ms)"].mean().tail(50)
        st.line_chart(df_img_avg)
    else:
        st.info("Log file is empty.")
else:
    st.info("No inference log found.")

# ==== 多平台性能对比 ====
st.markdown("### 🌍 Cross-Platform Performance Comparison")
uploaded_file = st.file_uploader("Upload Platform Results CSV", type="csv")

if uploaded_file:
    try:
        df_cmp = pd.read_csv(uploaded_file)
        st.dataframe(df_cmp)

        if "Model" in df_cmp.columns:
            selected_model = st.selectbox("Select Model", df_cmp["Model"].unique())
            df_model = df_cmp[df_cmp["Model"] == selected_model]
        else:
            df_model = df_cmp

        st.subheader("🕒 Inference Time (ms)")
        st.bar_chart(df_model.set_index("Platform")["Inference(ms)"])

        st.subheader("📸 FPS")
        st.bar_chart(df_model.set_index("Platform")["FPS"])

        if "Power(W)" in df_model.columns:
            st.subheader("🔌 Power Consumption (W)")
            st.bar_chart(df_model.set_index("Platform")["Power(W)"])

        if "LUT(%)" in df_model.columns and "BRAM(%)" in df_model.columns:
            st.subheader("📦 On-Chip Resource Usage (FPGA)")
            st.line_chart(df_model.set_index("Platform")[["LUT(%)", "BRAM(%)"]])

    except Exception as e:
        st.error(f"Error reading platform CSV: {e}")
